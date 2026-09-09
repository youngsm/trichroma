import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray as ga
from pycuda import characterize

from chroma.geometry import standard_wavelengths
from chroma.gpu.tools import get_cu_module, get_cu_source, cuda_options, \
    chunk_iterator, format_array, format_size, to_uint3, to_float3, \
    make_gpu_struct, mapped_empty, Mapped

from chroma.log import logger

class GPUGeometry(object):
    def __init__(self, geometry, wavelengths=None, times=None, print_usage=False, min_free_gpu_mem=300e6):
        if wavelengths is None:
            wavelengths = standard_wavelengths

        try:
            wavelength_step = np.unique(np.diff(wavelengths)).item()
        except ValueError:
            raise ValueError('wavelengths must be equally spaced apart.')
            
        if times is None:
            time_step = 0.05
            times = np.arange(0,1000,time_step)
        else:
            try:
                time_step = np.unique(np.diff(times)).item()
            except ValueError:
                raise ValueError('times must be equally spaced apart.')

        geometry_source = get_cu_source('geometry_types.h')
        material_struct_size = characterize.sizeof('Material', geometry_source)
        surface_struct_size = characterize.sizeof('Surface', geometry_source)
        dichroicprops_struct_size = characterize.sizeof('DichroicProps', geometry_source)
        angularprops_struct_size = characterize.sizeof('AngularProps', geometry_source)
        geometry_struct_size = characterize.sizeof('Geometry', geometry_source)
        wireplane_struct_size = characterize.sizeof('WirePlane', geometry_source)

        self.material_data = []
        self.material_ptrs = []
        materials_list = list(geometry.unique_materials)

        def interp_material_property(wavelengths, property):
            assert property is not None, 'property must not be None'
            # note that it is essential that the material properties be
            # interpolated linearly. this fact is used in the propagation
            # code to guarantee that probabilities still sum to one.
            return np.interp(wavelengths, property[:,0], property[:,1]).astype(np.float32)

        for i in range(len(materials_list)):
            material = materials_list[i]

            if material is None:
                raise Exception('one or more triangles is missing a material.')
            try:
                refractive_index = interp_material_property(wavelengths, material.refractive_index)
                refractive_index_gpu = ga.to_gpu(refractive_index)
                absorption_length = interp_material_property(wavelengths, material.absorption_length)
                absorption_length_gpu = ga.to_gpu(absorption_length)
                scattering_length = interp_material_property(wavelengths, material.scattering_length)
                scattering_length_gpu = ga.to_gpu(scattering_length)
            except Exception as e:
                print('Error with material %s: %s' % (material.name, e))
                raise e
            num_comp = len(material.comp_reemission_prob)
            comp_reemission_prob_gpu = [ga.to_gpu(interp_material_property(wavelengths, component)) for component in material.comp_reemission_prob]
            self.material_data.append(comp_reemission_prob_gpu)
            comp_reemission_prob_gpu = np.uint64(0) if len(comp_reemission_prob_gpu) == 0 else make_gpu_struct(8*len(comp_reemission_prob_gpu), comp_reemission_prob_gpu)
            assert num_comp == len(material.comp_reemission_wvl_cdf), 'component arrays must be same length'
            comp_reemission_wvl_cdf_gpu = [ga.to_gpu(interp_material_property(wavelengths, component)) for component in material.comp_reemission_wvl_cdf]
            self.material_data.append(comp_reemission_wvl_cdf_gpu)
            comp_reemission_wvl_cdf_gpu = np.uint64(0) if len(comp_reemission_wvl_cdf_gpu) == 0 else make_gpu_struct(8*len(comp_reemission_wvl_cdf_gpu), comp_reemission_wvl_cdf_gpu)
            assert num_comp == len(material.comp_reemission_time_cdf), 'component arrays must be same length'
            comp_reemission_time_cdf_gpu = [ga.to_gpu(interp_material_property(times, component)) for component in material.comp_reemission_time_cdf]
            self.material_data.append(comp_reemission_time_cdf_gpu)
            comp_reemission_time_cdf_gpu = np.uint64(0) if len(comp_reemission_time_cdf_gpu) == 0 else make_gpu_struct(8*len(comp_reemission_time_cdf_gpu), comp_reemission_time_cdf_gpu)
            assert num_comp == len(material.comp_absorption_length), 'component arrays must be same length'
            comp_absorption_length_gpu = [ga.to_gpu(interp_material_property(wavelengths, component)) for component in material.comp_absorption_length]
            self.material_data.append(comp_absorption_length_gpu)
            comp_absorption_length_gpu = np.uint64(0) if len(comp_absorption_length_gpu) == 0 else make_gpu_struct(8*len(comp_absorption_length_gpu), comp_absorption_length_gpu)

            self.material_data.append(refractive_index_gpu)
            self.material_data.append(absorption_length_gpu)
            self.material_data.append(scattering_length_gpu)
            self.material_data.append(comp_reemission_prob_gpu)
            self.material_data.append(comp_reemission_wvl_cdf_gpu)
            self.material_data.append(comp_reemission_time_cdf_gpu)
            self.material_data.append(comp_absorption_length_gpu)

            material_gpu = \
                make_gpu_struct(material_struct_size,
                                [refractive_index_gpu, absorption_length_gpu,
                                 scattering_length_gpu,
                                 comp_reemission_prob_gpu,
                                 comp_reemission_wvl_cdf_gpu,
                                 comp_reemission_time_cdf_gpu,
                                 comp_absorption_length_gpu,
                                 np.uint32(num_comp),
                                 np.uint32(len(wavelengths)),
                                 np.float32(wavelength_step),
                                 np.float32(wavelengths[0]),
                                 np.uint32(len(times)),
                                 np.float32(time_step),
                                 np.float32(times[0])])

            self.material_ptrs.append(material_gpu)

        # include any materials referenced only by analytic wireplanes
        plane_descs_for_materials = getattr(geometry, 'wireplanes', None) or []
        for desc in plane_descs_for_materials:
            for mat in (desc.get('material_inner', None), desc.get('material_outer', None)):
                if mat is None:
                    continue
                if mat in materials_list:
                    continue
                # append new material to lists and GPU buffers
                try:
                    refractive_index = interp_material_property(wavelengths, mat.refractive_index)
                    refractive_index_gpu = ga.to_gpu(refractive_index)
                    absorption_length = interp_material_property(wavelengths, mat.absorption_length)
                    absorption_length_gpu = ga.to_gpu(absorption_length)
                    scattering_length = interp_material_property(wavelengths, mat.scattering_length)
                    scattering_length_gpu = ga.to_gpu(scattering_length)
                except Exception as e:
                    print('Error with material %s: %s' % (mat.name, e))
                    raise e
                num_comp = len(mat.comp_reemission_prob)
                comp_reemission_prob_gpu = [ga.to_gpu(interp_material_property(wavelengths, component)) for component in mat.comp_reemission_prob]
                self.material_data.append(comp_reemission_prob_gpu)
                comp_reemission_prob_gpu = np.uint64(0) if len(comp_reemission_prob_gpu) == 0 else make_gpu_struct(8*len(comp_reemission_prob_gpu), comp_reemission_prob_gpu)
                assert num_comp == len(mat.comp_reemission_wvl_cdf), 'component arrays must be same length'
                comp_reemission_wvl_cdf_gpu = [ga.to_gpu(interp_material_property(wavelengths, component)) for component in mat.comp_reemission_wvl_cdf]
                self.material_data.append(comp_reemission_wvl_cdf_gpu)
                comp_reemission_wvl_cdf_gpu = np.uint64(0) if len(comp_reemission_wvl_cdf_gpu) == 0 else make_gpu_struct(8*len(comp_reemission_wvl_cdf_gpu), comp_reemission_wvl_cdf_gpu)
                assert num_comp == len(mat.comp_reemission_time_cdf), 'component arrays must be same length'
                comp_reemission_time_cdf_gpu = [ga.to_gpu(interp_material_property(times, component)) for component in mat.comp_reemission_time_cdf]
                self.material_data.append(comp_reemission_time_cdf_gpu)
                comp_reemission_time_cdf_gpu = np.uint64(0) if len(comp_reemission_time_cdf_gpu) == 0 else make_gpu_struct(8*len(comp_reemission_time_cdf_gpu), comp_reemission_time_cdf_gpu)
                assert num_comp == len(mat.comp_absorption_length), 'component arrays must be same length'
                comp_absorption_length_gpu = [ga.to_gpu(interp_material_property(wavelengths, component)) for component in mat.comp_absorption_length]
                self.material_data.append(comp_absorption_length_gpu)
                comp_absorption_length_gpu = np.uint64(0) if len(comp_absorption_length_gpu) == 0 else make_gpu_struct(8*len(comp_absorption_length_gpu), comp_absorption_length_gpu)

                self.material_data.extend([refractive_index_gpu, absorption_length_gpu, scattering_length_gpu, comp_reemission_prob_gpu, comp_reemission_wvl_cdf_gpu, comp_reemission_time_cdf_gpu, comp_absorption_length_gpu])

                material_gpu = make_gpu_struct(material_struct_size,
                                              [refractive_index_gpu, absorption_length_gpu,
                                               scattering_length_gpu,
                                               comp_reemission_prob_gpu,
                                               comp_reemission_wvl_cdf_gpu,
                                               comp_reemission_time_cdf_gpu,
                                               comp_absorption_length_gpu,
                                               np.uint32(num_comp),
                                               np.uint32(len(wavelengths)),
                                               np.float32(wavelength_step),
                                               np.float32(wavelengths[0]),
                                               np.uint32(len(times)),
                                               np.float32(time_step),
                                               np.float32(times[0])])

                self.material_ptrs.append(material_gpu)
                materials_list.append(mat)

        self.material_pointer_array = make_gpu_struct(8*len(self.material_ptrs), self.material_ptrs)

        self.surface_data = []
        self.surface_ptrs = []
        surfaces_list = list(geometry.unique_surfaces)

        for i in range(len(surfaces_list)):
            surface = surfaces_list[i]

            if surface is None:
                # need something to copy to the surface array struct
                # that is the same size as a 64-bit pointer.
                # this pointer will never be used by the simulation.
                self.surface_ptrs.append(np.uint64(0))
                continue

            detect = interp_material_property(wavelengths, surface.detect)
            detect_gpu = ga.to_gpu(detect)
            absorb = interp_material_property(wavelengths, surface.absorb)
            absorb_gpu = ga.to_gpu(absorb)
            reemit = interp_material_property(wavelengths, surface.reemit)
            reemit_gpu = ga.to_gpu(reemit)
            reflect_diffuse = interp_material_property(wavelengths, surface.reflect_diffuse)
            reflect_diffuse_gpu = ga.to_gpu(reflect_diffuse)
            reflect_specular = interp_material_property(wavelengths, surface.reflect_specular)
            reflect_specular_gpu = ga.to_gpu(reflect_specular)
            eta = interp_material_property(wavelengths, surface.eta)
            eta_gpu = ga.to_gpu(eta)
            k = interp_material_property(wavelengths, surface.k)
            k_gpu = ga.to_gpu(k)
            reemission_cdf = interp_material_property(wavelengths, surface.reemission_cdf)
            reemission_cdf_gpu = ga.to_gpu(reemission_cdf)
            
            if surface.dichroic_props:
                props = surface.dichroic_props
                transmit_pointers = []
                reflect_pointers = []
                angles_gpu = ga.to_gpu(np.asarray(props.angles,dtype=np.float32))
                self.surface_data.append(angles_gpu)
                
                for i,angle in enumerate(props.angles):
                    dichroic_reflect = interp_material_property(wavelengths, props.dichroic_reflect[i])
                    dichroic_reflect_gpu = ga.to_gpu(dichroic_reflect)
                    self.surface_data.append(dichroic_reflect_gpu)
                    reflect_pointers.append(dichroic_reflect_gpu)
                    
                    dichroic_transmit = interp_material_property(wavelengths, props.dichroic_transmit[i])
                    dichroic_transmit_gpu = ga.to_gpu(dichroic_transmit)
                    self.surface_data.append(dichroic_transmit_gpu)
                    transmit_pointers.append(dichroic_transmit_gpu)
                
                reflect_arr_gpu = make_gpu_struct(8*len(reflect_pointers),reflect_pointers)
                self.surface_data.append(reflect_arr_gpu)
                transmit_arr_gpu = make_gpu_struct(8*len(transmit_pointers), transmit_pointers)
                self.surface_data.append(transmit_arr_gpu)
                dichroic_props = make_gpu_struct(dichroicprops_struct_size,[angles_gpu,reflect_arr_gpu,transmit_arr_gpu,np.uint32(len(props.angles))])
            else:
                dichroic_props = np.uint64(0) #NULL
            
            if surface.angular_props:
                props = surface.angular_props
                angles_gpu = ga.to_gpu(np.asarray(props.angles, dtype=np.float32))
                transmit_gpu = ga.to_gpu(np.asarray(props.transmit, dtype=np.float32))
                reflect_spec_gpu = ga.to_gpu(np.asarray(props.reflect_specular, dtype=np.float32))
                reflect_diff_gpu = ga.to_gpu(np.asarray(props.reflect_diffuse, dtype=np.float32))
                
                self.surface_data.extend([angles_gpu, transmit_gpu, reflect_spec_gpu, reflect_diff_gpu])
                angular_props = make_gpu_struct(angularprops_struct_size, 
                                               [angles_gpu, transmit_gpu, reflect_spec_gpu, reflect_diff_gpu, 
                                                np.uint32(len(props.angles))])
            else:
                angular_props = np.uint64(0)  # NULL
            
            self.surface_data.append(detect_gpu)
            self.surface_data.append(absorb_gpu)
            self.surface_data.append(reemit_gpu)
            self.surface_data.append(reflect_diffuse_gpu)
            self.surface_data.append(reflect_specular_gpu)
            self.surface_data.append(reemission_cdf_gpu)
            self.surface_data.append(eta_gpu)
            self.surface_data.append(k_gpu)
            self.surface_data.append(dichroic_props)
            self.surface_data.append(angular_props)
            
            surface_gpu = \
                make_gpu_struct(surface_struct_size,
                                [detect_gpu, absorb_gpu, reemit_gpu,
                                 reflect_diffuse_gpu,reflect_specular_gpu,
                                 eta_gpu, k_gpu, reemission_cdf_gpu,
                                 dichroic_props,
                                 angular_props,
                                 np.uint32(surface.model),
                                 np.uint32(len(wavelengths)),
                                 np.uint32(surface.transmissive),
                                 np.float32(wavelength_step),
                                 np.float32(wavelengths[0]),
                                 np.float32(surface.thickness)])

            self.surface_ptrs.append(surface_gpu)

        # include any surfaces referenced only by analytic wireplanes
        for desc in plane_descs_for_materials:
            surface = desc.get('surface', None)
            if surface is None:
                continue
            if surface in surfaces_list:
                continue
            detect = interp_material_property(wavelengths, surface.detect)
            detect_gpu = ga.to_gpu(detect)
            absorb = interp_material_property(wavelengths, surface.absorb)
            absorb_gpu = ga.to_gpu(absorb)
            reemit = interp_material_property(wavelengths, surface.reemit)
            reemit_gpu = ga.to_gpu(reemit)
            reflect_diffuse = interp_material_property(wavelengths, surface.reflect_diffuse)
            reflect_diffuse_gpu = ga.to_gpu(reflect_diffuse)
            reflect_specular = interp_material_property(wavelengths, surface.reflect_specular)
            reflect_specular_gpu = ga.to_gpu(reflect_specular)
            eta = interp_material_property(wavelengths, surface.eta)
            eta_gpu = ga.to_gpu(eta)
            k = interp_material_property(wavelengths, surface.k)
            k_gpu = ga.to_gpu(k)
            reemission_cdf = interp_material_property(wavelengths, surface.reemission_cdf)
            reemission_cdf_gpu = ga.to_gpu(reemission_cdf)

            if surface.dichroic_props:
                props = surface.dichroic_props
                transmit_pointers = []
                reflect_pointers = []
                angles_gpu = ga.to_gpu(np.asarray(props.angles,dtype=np.float32))
                self.surface_data.append(angles_gpu)
                for i_ang,angle in enumerate(props.angles):
                    dichroic_reflect = interp_material_property(wavelengths, props.dichroic_reflect[i_ang])
                    dichroic_reflect_gpu = ga.to_gpu(dichroic_reflect)
                    self.surface_data.append(dichroic_reflect_gpu)
                    reflect_pointers.append(dichroic_reflect_gpu)
                    dichroic_transmit = interp_material_property(wavelengths, props.dichroic_transmit[i_ang])
                    dichroic_transmit_gpu = ga.to_gpu(dichroic_transmit)
                    self.surface_data.append(dichroic_transmit_gpu)
                    transmit_pointers.append(dichroic_transmit_gpu)
                reflect_arr_gpu = make_gpu_struct(8*len(reflect_pointers),reflect_pointers)
                self.surface_data.append(reflect_arr_gpu)
                transmit_arr_gpu = make_gpu_struct(8*len(transmit_pointers), transmit_pointers)
                self.surface_data.append(transmit_arr_gpu)
                dichroic_props = make_gpu_struct(dichroicprops_struct_size,[angles_gpu,reflect_arr_gpu,transmit_arr_gpu,np.uint32(len(props.angles))])
            else:
                dichroic_props = np.uint64(0)

            if surface.angular_props:
                props = surface.angular_props
                angles_gpu = ga.to_gpu(np.asarray(props.angles, dtype=np.float32))
                transmit_gpu = ga.to_gpu(np.asarray(props.transmit, dtype=np.float32))
                reflect_spec_gpu = ga.to_gpu(np.asarray(props.reflect_specular, dtype=np.float32))
                reflect_diff_gpu = ga.to_gpu(np.asarray(props.reflect_diffuse, dtype=np.float32))
                self.surface_data.extend([angles_gpu, transmit_gpu, reflect_spec_gpu, reflect_diff_gpu])
                angular_props = make_gpu_struct(angularprops_struct_size, 
                                               [angles_gpu, transmit_gpu, reflect_spec_gpu, reflect_diff_gpu, 
                                                np.uint32(len(props.angles))])
            else:
                angular_props = np.uint64(0)

            self.surface_data.extend([detect_gpu, absorb_gpu, reemit_gpu, reflect_diffuse_gpu, reflect_specular_gpu, reemission_cdf_gpu, eta_gpu, k_gpu, dichroic_props, angular_props])
            surface_gpu = make_gpu_struct(surface_struct_size,
                                [detect_gpu, absorb_gpu, reemit_gpu,
                                 reflect_diffuse_gpu,reflect_specular_gpu,
                                 eta_gpu, k_gpu, reemission_cdf_gpu,
                                 dichroic_props,
                                 angular_props,
                                 np.uint32(surface.model),
                                 np.uint32(len(wavelengths)),
                                 np.uint32(surface.transmissive),
                                 np.float32(wavelength_step),
                                 np.float32(wavelengths[0]),
                                 np.float32(surface.thickness)])
            self.surface_ptrs.append(surface_gpu)
            surfaces_list.append(surface)

        self.surface_pointer_array = make_gpu_struct(8*len(self.surface_ptrs), self.surface_ptrs)

        # analytic wire-planes (optional)
        self.wireplane_ptrs = []
        self.wireplane_data = []
        # build lookups for indices including any extras
        material_lookup = dict(list(zip(materials_list, list(range(len(materials_list))))))
        surface_lookup = dict(list(zip(surfaces_list, list(range(len(surfaces_list))))))
        plane_descs = getattr(geometry, 'wireplanes', None)
        if plane_descs is None:
            plane_descs = []
        for desc in plane_descs:
            origin = ga.vec.make_float3(*np.asarray(desc['origin'], dtype=np.float32))
            u = ga.vec.make_float3(*np.asarray(desc['u'], dtype=np.float32))
            v = ga.vec.make_float3(*np.asarray(desc['v'], dtype=np.float32))
            pitch = np.float32(desc['pitch'])
            radius = np.float32(desc['radius'])
            umin = np.float32(desc['umin'])
            umax = np.float32(desc['umax'])
            vmin = np.float32(desc['vmin'])
            vmax = np.float32(desc['vmax'])
            v0 = np.float32(desc['v0'])
            surface = desc.get('surface', None)
            material_inner = desc.get('material_inner', None)
            material_outer = desc.get('material_outer', None)
            color = np.uint32(desc.get('color', 0))

            surface_idx = -1 if surface is None else int(surface_lookup.get(surface, -1))
            if material_outer is None or material_inner is None:
                # fall back to first entries (should not happen under normal use)
                material_outer_idx = 0
                material_inner_idx = 0
            else:
                material_outer_idx = int(material_lookup[material_outer])
                material_inner_idx = int(material_lookup[material_inner])

            plane_gpu = make_gpu_struct(
                wireplane_struct_size,
                [origin, u, v, pitch, radius, umin, umax, vmin, vmax, v0,
                 np.int32(surface_idx), np.int32(material_outer_idx), np.int32(material_inner_idx), color]
            )
            self.wireplane_ptrs.append(plane_gpu)

        if len(self.wireplane_ptrs) > 0:
            self.wireplane_pointer_array = make_gpu_struct(8*len(self.wireplane_ptrs), self.wireplane_ptrs)
        else:
            self.wireplane_pointer_array = np.uint64(0)  # NULL

        self.vertices = mapped_empty(shape=len(geometry.mesh.vertices),
                                     dtype=ga.vec.float3,
                                     write_combined=True)
        self.triangles = mapped_empty(shape=len(geometry.mesh.triangles),
                                      dtype=ga.vec.uint3,
                                      write_combined=True)
        self.vertices[:] = to_float3(geometry.mesh.vertices)
        self.triangles[:] = to_uint3(geometry.mesh.triangles)
        
        self.world_origin = ga.vec.make_float3(*geometry.bvh.world_coords.world_origin)
        self.world_scale = np.float32(geometry.bvh.world_coords.world_scale)

        material_codes = (((geometry.material1_index & 0xff) << 24) |
                          ((geometry.material2_index & 0xff) << 16) |
                          ((geometry.surface_index & 0xff) << 8)).astype(np.uint32)
        self.material_codes = ga.to_gpu(material_codes)
        colors = geometry.colors.astype(np.uint32)
        self.colors = ga.to_gpu(colors)
        self.solid_id_map = ga.to_gpu(geometry.solid_id.astype(np.uint32))

        # Limit memory usage by splitting BVH into on and off-GPU parts
        gpu_free, gpu_total = cuda.mem_get_info()
        # node_array_usage = geometry.bvh.nodes.nbytes

        # Figure out how many elements we can fit on the GPU,
        # but no fewer than 100 elements, and no more than the number of actual nodes
        n_nodes = len(geometry.bvh.nodes)
        split_index = min(
            max(int((gpu_free - min_free_gpu_mem) / geometry.bvh.nodes.itemsize),100),
            n_nodes
            )
        
        self.nodes = ga.to_gpu(geometry.bvh.nodes[:split_index])
        n_extra = max(1, (n_nodes - split_index)) # forbid zero size
        self.extra_nodes = mapped_empty(shape=n_extra,
                                        dtype=geometry.bvh.nodes.dtype,
                                        write_combined=True)
        if split_index < n_nodes:
            logger.info('Splitting BVH between GPU and CPU memory at node %d' % split_index)
            self.extra_nodes[:] = geometry.bvh.nodes[split_index:]

        # See if there is enough memory to put the and/ortriangles back on the GPU
        gpu_free, gpu_total = cuda.mem_get_info()
        if self.triangles.nbytes < (gpu_free - min_free_gpu_mem):
            self.triangles = ga.to_gpu(self.triangles)
            logger.info('Optimization: Sufficient memory to move triangles onto GPU')

        gpu_free, gpu_total = cuda.mem_get_info()
        if self.vertices.nbytes < (gpu_free - min_free_gpu_mem):
            self.vertices = ga.to_gpu(self.vertices)
            logger.info('Optimization: Sufficient memory to move vertices onto GPU')

        # analytic wire-plane descriptors for fast intersection
        wireplanes_ptr = np.uint64(0)
        nwireplanes = 0
        if hasattr(geometry, 'wireplanes') and geometry.wireplanes is not None and len(geometry.wireplanes) > 0:
            # build lookup for materials/surfaces in this geometry
            mat_lookup = {m: i for i, m in enumerate(geometry.unique_materials)}
            surf_lookup = {s: i for i, s in enumerate(geometry.unique_surfaces)}

            members = []
            for wp in geometry.wireplanes:
                get = (lambda key, default=None: (wp.get(key, default) if isinstance(wp, dict) else getattr(wp, key, default)))

                origin = np.asarray(get('origin'), dtype=np.float32)
                u = np.asarray(get('u'), dtype=np.float32)
                v = np.asarray(get('v'), dtype=np.float32)
                pitch = np.float32(get('pitch'))
                radius = np.float32(get('radius'))
                umin = np.float32(get('umin', -1e9))
                umax = np.float32(get('umax', +1e9))
                vmin = np.float32(get('vmin', -1e9))
                vmax = np.float32(get('vmax', +1e9))
                v0 = np.float32(get('v0', 0.0))

                # surface/material indices can be specified directly or via objects
                surf_idx = get('surface_index', None)
                if surf_idx is None:
                    surf_obj = get('surface', None)
                    surf_idx = surf_lookup.get(surf_obj, None)
                if surf_idx is None or int(surf_idx) < 0 or int(surf_idx) >= len(geometry.unique_surfaces):
                    raise ValueError('WirePlane surface unresolved: provide surface_index or ensure surface object is in geometry.unique_surfaces')
                surf_idx = np.int32(int(surf_idx))

                m_out_idx = get('material_outer_index', None)
                if m_out_idx is None:
                    mo = get('material_outer', None)
                    m_out_idx = mat_lookup.get(mo, None)
                if m_out_idx is None or int(m_out_idx) < 0 or int(m_out_idx) >= len(geometry.unique_materials):
                    raise ValueError('WirePlane material_outer unresolved: provide material_outer_index or ensure material object is in geometry.unique_materials')
                m_out_idx = np.int32(int(m_out_idx))

                m_in_idx = get('material_inner_index', None)
                if m_in_idx is None:
                    mi = get('material_inner', None)
                    m_in_idx = mat_lookup.get(mi, None)
                if m_in_idx is None or int(m_in_idx) < 0 or int(m_in_idx) >= len(geometry.unique_materials):
                    raise ValueError('WirePlane material_inner unresolved: provide material_inner_index or ensure material object is in geometry.unique_materials')
                m_in_idx = np.int32(int(m_in_idx))

                color = np.uint32(get('color', 0))

                members.extend([
                    ga.vec.make_float3(*origin),
                    ga.vec.make_float3(*u),
                    ga.vec.make_float3(*v),
                    pitch,
                    radius,
                    umin, umax, vmin, vmax,
                    v0,
                    surf_idx,
                    m_out_idx,
                    m_in_idx,
                    color
                ])

            wireplanes_ptr = make_gpu_struct(wireplane_struct_size*len(geometry.wireplanes), members)
            nwireplanes = len(geometry.wireplanes)

        self.gpudata = make_gpu_struct(geometry_struct_size,
                                       [Mapped(self.vertices), 
                                        Mapped(self.triangles),
                                        self.material_codes,
                                        self.colors, self.nodes,
                                        Mapped(self.extra_nodes),
                                        self.material_pointer_array,
                                        self.surface_pointer_array,
                                        self.wireplane_pointer_array,
                                        self.world_origin,
                                        self.world_scale,
                                        np.int32(len(self.nodes)),
                                        np.int32(len(self.wireplane_ptrs))])

        self.geometry = geometry

        if print_usage:
            self.print_device_usage()
        logger.info(self.device_usage_str())

    def device_usage_str(self):
        '''Returns a formatted string displaying the memory usage.'''
        s = 'device usage:\n'
        s += '-'*10 + '\n'
        #s += format_array('vertices', self.vertices) + '\n'
        #s += format_array('triangles', self.triangles) + '\n'
        s += format_array('nodes', self.nodes) + '\n'
        s += '%-15s %6s %6s' % ('total', '', format_size(self.nodes.nbytes)) + '\n'
        s += '-'*10 + '\n'
        free, total = cuda.mem_get_info()
        s += '%-15s %6s %6s' % ('device total', '', format_size(total)) + '\n'
        s += '%-15s %6s %6s' % ('device used', '', format_size(total-free)) + '\n'
        s += '%-15s %6s %6s' % ('device free', '', format_size(free)) + '\n'
        return s

    def print_device_usage(self):
        print(self.device_usage_str())
        print() 

    def reset_colors(self):
        self.colors.set_async(self.geometry.colors.astype(np.uint32))

    def color_solids(self, solid_hit, colors, nblocks_per_thread=64,
                     max_blocks=1024):
        solid_hit_gpu = ga.to_gpu(np.array(solid_hit, dtype=np.bool))
        solid_colors_gpu = ga.to_gpu(np.array(colors, dtype=np.uint32))

        module = get_cu_module('mesh.h', options=cuda_options)
        color_solids = module.get_function('color_solids')

        for first_triangle, triangles_this_round, blocks in \
                chunk_iterator(self.triangles.size, nblocks_per_thread,
                               max_blocks):
            color_solids(np.int32(first_triangle),
                         np.int32(triangles_this_round), self.solid_id_map,
                         solid_hit_gpu, solid_colors_gpu, self.gpudata,
                         block=(nblocks_per_thread,1,1), 
                         grid=(blocks,1))
