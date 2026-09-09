// Spectral adjoint camera. Photon maps contain every deposited optical packet.
// Compose after physics.wgsl + deposition.wgsl; this adds no extra transport law.
struct CameraParameters {eye:vec4<f32>,look_at:vec4<f32>,image:vec4<u32>,options:vec4<f32>}
@group(0) @binding(8) var<uniform> camera:CameraParameters;
@group(0) @binding(9) var camera_image:texture_storage_2d<rgba16float,write>;
@group(0) @binding(10) var<storage,read_write> accumulation:array<vec4<f32>>;
fn map_wall(face:u32,uv:vec2<f32>,bin:u32)->f32{
    let coordinate=clamp(uv*f32(WALL_RES)-.5,vec2<f32>(0.),vec2<f32>(f32(WALL_RES-1u)));let lo=vec2<u32>(coordinate);let hi=min(lo+1u,vec2<u32>(WALL_RES-1u));let a=fract(coordinate);
    let p00=bitcast<f32>(atomicLoad(&wall_map[((face*WALL_RES+lo.y)*WALL_RES+lo.x)*WAVELENGTH_BINS+bin]));
    let p10=bitcast<f32>(atomicLoad(&wall_map[((face*WALL_RES+lo.y)*WALL_RES+hi.x)*WAVELENGTH_BINS+bin]));
    let p01=bitcast<f32>(atomicLoad(&wall_map[((face*WALL_RES+hi.y)*WALL_RES+lo.x)*WAVELENGTH_BINS+bin]));
    let p11=bitcast<f32>(atomicLoad(&wall_map[((face*WALL_RES+hi.y)*WALL_RES+hi.x)*WAVELENGTH_BINS+bin]));
    return mix(mix(p00,p10,a.x),mix(p01,p11,a.x),a.y);
}
fn map_fill_wall(face:u32,uv:vec2<f32>,bin:u32)->f32{
    let coordinate=clamp(uv*f32(FILL_RES)-.5,vec2<f32>(0.),vec2<f32>(f32(FILL_RES-1u)));let lo=vec2<u32>(coordinate);let hi=min(lo+1u,vec2<u32>(FILL_RES-1u));let a=fract(coordinate);
    let p00=bitcast<f32>(atomicLoad(&fill_wall_map[((face*FILL_RES+lo.y)*FILL_RES+lo.x)*WAVELENGTH_BINS+bin]));
    let p10=bitcast<f32>(atomicLoad(&fill_wall_map[((face*FILL_RES+lo.y)*FILL_RES+hi.x)*WAVELENGTH_BINS+bin]));
    let p01=bitcast<f32>(atomicLoad(&fill_wall_map[((face*FILL_RES+hi.y)*FILL_RES+lo.x)*WAVELENGTH_BINS+bin]));
    let p11=bitcast<f32>(atomicLoad(&fill_wall_map[((face*FILL_RES+hi.y)*FILL_RES+hi.x)*WAVELENGTH_BINS+bin]));
    return mix(mix(p00,p10,a.x),mix(p01,p11,a.x),a.y);
}
fn map_emission(face:u32,hemisphere:u32,uv:vec2<f32>,bin:u32)->f32{
    let coordinate=clamp(uv*f32(EMISSION_RES)-.5,vec2<f32>(0.),vec2<f32>(f32(EMISSION_RES-1u)));let lo=vec2<u32>(coordinate);let hi=min(lo+1u,vec2<u32>(EMISSION_RES-1u));let a=fract(coordinate);let f=face*2u+hemisphere;
    let p00=bitcast<f32>(atomicLoad(&emission_map[((f*EMISSION_RES+lo.y)*EMISSION_RES+lo.x)*WAVELENGTH_BINS+bin]));
    let p10=bitcast<f32>(atomicLoad(&emission_map[((f*EMISSION_RES+lo.y)*EMISSION_RES+hi.x)*WAVELENGTH_BINS+bin]));
    let p01=bitcast<f32>(atomicLoad(&emission_map[((f*EMISSION_RES+hi.y)*EMISSION_RES+lo.x)*WAVELENGTH_BINS+bin]));
    let p11=bitcast<f32>(atomicLoad(&emission_map[((f*EMISSION_RES+hi.y)*EMISSION_RES+hi.x)*WAVELENGTH_BINS+bin]));
    return mix(mix(p00,p10,a.x),mix(p01,p11,a.x),a.y);
}
fn cell_scattering_moment(cell:vec3<u32>,bin:u32,e:vec3<f32>)->f32{
    let base=(((cell.z*VOLUME_Y+cell.y)*VOLUME_X+cell.x)*WAVELENGTH_BINS+bin)*6u;
    let xx=bitcast<f32>(atomicLoad(&volume_map[base]));let yy=bitcast<f32>(atomicLoad(&volume_map[base+1u]));let zz=bitcast<f32>(atomicLoad(&volume_map[base+2u]));
    if(xx+yy+zz==0.){return 0.;}
    let xy=bitcast<f32>(atomicLoad(&volume_map[base+3u]));let xz=bitcast<f32>(atomicLoad(&volume_map[base+4u]));let yz=bitcast<f32>(atomicLoad(&volume_map[base+5u]));
    return max(0.,e.x*e.x*xx+e.y*e.y*yy+e.z*e.z*zz+2.*(e.x*e.y*xy+e.x*e.z*xz+e.y*e.z*yz));
}
fn camera_scattering_source(position:vec3<f32>,bin:u32,polarization:vec3<f32>)->f32{
    // Continuous tent reconstruction of cell-centered collision moments.
    // This adds finite-resolution smoothing bias, without N-dependent gain.
    let coordinate=clamp((position+ROOM)/(2.*ROOM)*vec3<f32>(volume_shape())-.5,vec3<f32>(0.),vec3<f32>(volume_shape())-1.);
    let lo=vec3<u32>(coordinate);let hi=min(lo+1u,volume_shape()-1u);let a=fract(coordinate);
    var moment=0.;
    for(var z=0u;z<2u;z++){for(var y=0u;y<2u;y++){for(var x=0u;x<2u;x++){
        let cell=vec3<u32>(select(lo.x,hi.x,x==1u),select(lo.y,hi.y,y==1u),select(lo.z,hi.z,z==1u));
        let weight=select(1.-a.x,a.x,x==1u)*select(1.-a.y,a.y,y==1u)*select(1.-a.z,a.z,z==1u);
        moment+=weight*cell_scattering_moment(cell,bin,polarization);
    }}}
    // Random transverse camera polarization with weight2 is an unbiased
    // adjoint estimator of the unpolarized3/(8pi)*(trM-omega M omega) source.
    return moment*(3./(4.*3.141592653589793))/(f32(config.counts.x)*VOXEL_VOLUME_MM3);
}
struct VolumeIntegral {radiance:f32,transmittance:f32}
fn integrate_volume(origin:vec3<f32>,direction:vec3<f32>,polarization:vec3<f32>,distance:f32,bin:u32,material:u32,wavelength:f32)->VolumeIntegral{
    let scatter_length=interpolate(16u,material,wavelength);let absorb_length=interpolate(15u,material,wavelength);
    let sigma=select(0.,1./scatter_length,scatter_length<1e30)+select(0.,1./absorb_length,absorb_length<1e30);
    if(sigma<=0.){return VolumeIntegral(0.,1.);}
    var radiance=0.;var transmission=1.;var t=0.;
    let cell_size=2.*ROOM/vec3<f32>(volume_shape());
    for(var i=0u;i<256u;i++){
        if(t>=distance||transmission<1e-7){break;}
        let position=origin+(t+1e-4)*direction;
        let cell=vec3<u32>(clamp((position+ROOM)/cell_size,vec3<f32>(0.),vec3<f32>(volume_shape())-1.));
        let far=select(vec3<f32>(cell),vec3<f32>(cell)+1.,direction>vec3<f32>(0.))*cell_size-ROOM;
        let nonzero=abs(direction)>vec3<f32>(1e-30);
        let raw_bound=(far-origin)/select(vec3<f32>(1.),direction,nonzero);
        let bound=select(vec3<f32>(1e30),raw_bound,nonzero);
        let next=min(distance,max(t+1e-4,min(min(bound.x,bound.y),bound.z)));
        let length=next-t;let attenuation=exp(-sigma*length);
        // Use a stable series for short steps to avoid subtracting near-one values.
        let optical_depth=sigma*length;
        let integral=select((1.-attenuation)/sigma,length*(1.-.5*optical_depth+optical_depth*optical_depth/6.),optical_depth<.001);
        if(scatter_length<1e30){radiance+=transmission*camera_scattering_source(origin+direction*(t+.5*length),bin,polarization)*integral;}
        transmission*=attenuation;t=next;
    }
    return VolumeIntegral(radiance,transmission);
}
fn fill_emission(position:vec3<f32>,bin:u32)->f32{
    if(f(tables[41u]+85u)>.5){
        // Six equal-area wall panels share the original ceiling-light power.
        let gap=abs(abs(position)-ROOM);
        let axis=select(select(2u,1u,gap.y<gap.z),0u,gap.x<min(gap.y,gap.z));
        let u_axis=select(0u,1u,axis==0u);let v_axis=select(2u,1u,axis==2u);
        if(gap[axis]>.01||abs(position[u_axis])>f(tables[41u]+67u)||abs(position[v_axis])>f(tables[41u]+68u)){return 0.;}
        return f(tables[41u]+bin)/6.;
    }
    let center=v3(tables[41u]+64u);let delta=position-center;
    if(abs(delta.z)>.01||abs(delta.x)>f(tables[41u]+67u)||abs(delta.y)>f(tables[41u]+68u)){return 0.;}
    return f(tables[41u]+bin);
}
// Chroma's original compressed BVH and threaded escape links are reused.
// Leaf geometry and triangle IDs are unchanged; padding is only for pruning.
fn camera_box_hit(lo:vec3<f32>,hi:vec3<f32>,origin:vec3<f32>,direction:vec3<f32>,limit:f32)->bool{
    var near=0.;var far=limit;
    for(var axis=0u;axis<3u;axis++){
        if(direction[axis]==0.){if(origin[axis]<lo[axis]||origin[axis]>hi[axis]){return false;}}
        else{let a=(lo[axis]-origin[axis])/direction[axis];let b=(hi[axis]-origin[axis])/direction[axis];near=max(near,min(a,b));far=min(far,max(a,b));}
    }
    return near<=far;
}
fn nearest_camera_bvh(origin:vec3<f32>,direction:vec3<f32>,previous:u32,clipped:bool)->Boundary{
    var best=Boundary(END,1e30);var index=0u;var visits=0u;
    loop{
        if(index==END){break;}
        if(index>=tables[47u]||visits>=tables[47u]){return Boundary(END,1e30);}
        visits++;
        let base=tables[45u]+4u*index;let packed=vec3<u32>(tables[base],tables[base+1u],tables[base+2u]);
        let offset=v3(tables[41u]+88u);let scale=f(tables[41u]+91u);
        let lo=offset+(vec3<f32>(packed&vec3<u32>(65535u))-1.)*scale;
        let hi=offset+(vec3<f32>(packed>>vec3<u32>(16u))+1.)*scale;
        let word=tables[base+3u];let child=word&0x0fffffffu;let successor=tables[tables[46u]+index];
        if(!camera_box_hit(lo,hi,origin,direction,best.distance)){index=successor;continue;}
        if((word>>28u)!=0u){index=child;continue;}
        index=successor;if(child==previous){continue;}
        let vertex=tables[8u]+9u*child;let a=v3(vertex);let e1=v3(vertex+3u)-a;let e2=v3(vertex+6u)-a;
        let h=cross(direction,e2);let determinant=dot(e1,h);
        if(abs(determinant)<=1.1920928955078125e-7){continue;}
        let reciprocal=1./determinant;let delta=origin-a;let u=reciprocal*dot(delta,h);let q=cross(delta,e1);
        let v=reciprocal*dot(direction,q);let distance=reciprocal*dot(e2,q);
        if(u< -1e-6||u>1.000001||v< -1e-6||u+v>1.000001||distance<=1e-6||distance>best.distance||(distance==best.distance&&child>=best.triangle)){continue;}
        if(clipped&&tables[tables[42u]+child]!=0u){
            let point=origin+distance*direction;
            if((dot(point,v3(tables[41u]+80u))-f(tables[41u]+83u))*f(tables[41u]+84u)<0.){continue;}
        }
        best=Boundary(child,distance);
    }
    return best;
}
// Clipping is an explicitly non-radiometric inspection view of unchanged maps.
// Full transport and the uncut camera always use every original triangle.
fn nearest_camera_boundary(origin:vec3<f32>,direction:vec3<f32>,previous:u32)->Boundary{
    if(camera.options.x<.5||tables[42u]==0u){return nearest_boundary(origin,direction,previous);}
    if(tables[45u]!=0u){return nearest_camera_bvh(origin,direction,previous,true);}
    var best=Boundary(END,1e30);
    for(var tri=0u;tri<tables[1u];tri++){
        if(tri==previous){continue;}
        let base=tables[8u]+tri*9u;let a=v3(base);let e1=v3(base+3u)-a;let e2=v3(base+6u)-a;
        let h=cross(direction,e2);let determinant=dot(e1,h);
        if(abs(determinant)<=1.1920928955078125e-7){continue;}
        let reciprocal=1./determinant;let delta=origin-a;let u=reciprocal*dot(delta,h);let q=cross(delta,e1);
        let v=reciprocal*dot(direction,q);let distance=reciprocal*dot(e2,q);
        if(u< -1e-6||u>1.000001||v< -1e-6||u+v>1.000001||distance<=1e-6||distance>=best.distance){continue;}
        if(tables[tables[42u]+tri]!=0u){
            let point=origin+distance*direction;
            if((dot(point,v3(tables[41u]+80u))-f(tables[41u]+83u))*f(tables[41u]+84u)<0.){continue;}
        }
        best=Boundary(tri,distance);
    }
    return best;
}
fn spectral_camera(origin:vec3<f32>,initial_direction:vec3<f32>,bin:u32,pixel:u32)->f32{
    let wavelength=select(283.59375+7.1875*f32(bin),128.,bin==0u&&tables[42u]!=0u&&camera.options.y>.5);
    var direction=initial_direction;var position=origin;
    let rng_id=pixel^(bin*0x9e3779b9u)^(camera.image.z*0x85ebca6bu);
    var polarization=random_polarization(direction,random_uniform(rng_id,0x20000000u));
    var previous=END;var throughput=1.;var radiance=0.;
    for(var bounce=0u;bounce<64u;bounce++){
        let hit=nearest_camera_boundary(position,direction,previous);if(hit.triangle==END){break;}
        let tri=hit.triangle;let normal=v3(tables[9u]+tri*3u);let outward=dot(direction,normal)>0.;
        let m1=tables[tables[10u]+tri];let m2=tables[tables[11u]+tri];let incident=select(m2,m1,outward);let other=select(m1,m2,outward);
        let volume=integrate_volume(position,direction,polarization,hit.distance,bin,incident,wavelength);
        radiance+=throughput*volume.radiance;throughput*=volume.transmittance;
        position+=hit.distance*direction;
        let surface=bitcast<i32>(tables[tables[12u]+tri]);
        if(surface>=0&&tables[tables[18u]+u32(surface)]!=0u){
            let model=tables[tables[19u]+u32(surface)];
            if(model==0u&&interpolate(20u,u32(surface),wavelength)>.99&&(tables[42u]==0u||tables[tables[42u]+tri]==0u)){
                let face=face_index(normal);let energy=map_wall(face,face_uv(position,ROOM,face),bin);
                let irradiance=energy/(f32(config.counts.x)*face_area(ROOM,face,WALL_RES))+map_fill_wall(face,face_uv(position,ROOM,face),bin)/(f32(config.counts.x)*face_area(ROOM,face,FILL_RES));
                radiance+=throughput*(irradiance*f(tables[41u]+76u)/3.141592653589793+fill_emission(position,bin));break;
            }
            if(model==0u){
                // Default surfaces include the non-emissive photocathode and
                // calibrated PMT back reflector. Detection absorbs camera rays.
                let draw=random_uniform(rng_id,0x30000000u+bounce*32u+6u);
                let absorbed=interpolate(21u,u32(surface),wavelength)+interpolate(20u,u32(surface),wavelength);
                let diffuse=interpolate(22u,u32(surface),wavelength);let specular=interpolate(23u,u32(surface),wavelength);
                if(draw<absorbed){break;}
                if(draw<absorbed+diffuse){
                    direction=hemisphere(select(normal,-normal,outward),random_uniform(rng_id,0x30000000u+bounce*32u+14u),random_uniform(rng_id,0x30000000u+bounce*32u+15u),true);
                    polarization=random_polarization(direction,random_uniform(rng_id,0x30000000u+bounce*32u+16u));
                    previous=tri;position=boundary_origin(position,direction,tri);continue;
                }
                if(draw<absorbed+diffuse+specular){
                    direction-=2.*dot(direction,normal)*normal;polarization-=2.*dot(polarization,normal)*normal;
                    previous=tri;position=boundary_origin(position,direction,tri);continue;
                }
            }
            if(model==2u){
                let face=face_index(normal);let hemisphere=select(1u,0u,dot(-direction,normal)>=0.);
                var energy=0.;var area=1.;
                if(tables[43u]!=0u){
                    let chart=bitcast<i32>(tables[tables[43u]+tri]);
                    if(chart>=0){energy=bitcast<f32>(atomicLoad(&emission_map[(2u*u32(chart)+hemisphere)*WAVELENGTH_BINS+bin]));area=f(tables[44u]+tri);}
                }else{
                    energy=map_emission(face,hemisphere,face_uv(position-v3(tables[41u]+70u),v3(tables[41u]+73u),face),bin);
                    area=face_area(v3(tables[41u]+73u),face,EMISSION_RES);
                }
                radiance+=throughput*energy/(f32(config.counts.x)*area*6.283185307179586*max(abs(dot(direction,normal)),1e-6));
                throughput*=1.-interpolate(21u,u32(surface),wavelength);
            }
        }
        if(throughput<1e-8){break;}
        let n1=interpolate(14u,incident,wavelength);let n2=interpolate(14u,other,wavelength);
        let ray=fresnel(direction,polarization,select(normal,-normal,outward),n1,n2,rng_id,0x30000000u+bounce*32u);
        if(!ray.reflected){throughput*=n1*n1/(n2*n2);}
        direction=ray.direction;polarization=ray.polarization;previous=tri;position=boundary_origin(position,direction,tri);
    }
    return radiance;
}
fn srgb(value:vec3<f32>)->vec3<f32>{return select(12.92*value,1.055*pow(value,vec3<f32>(1./2.4))-.055,value>vec3<f32>(.0031308));}
@compute @workgroup_size(8,8)
fn render_camera(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_id) lane:vec3<u32>){
    let block=group.x+camera.image.w;let columns=(camera.image.x+7u)/8u;
    let xy=vec2<u32>(block%columns,block/columns)*8u+lane.xy;if(any(xy>=camera.image.xy)){return;}
    let pixel=xy.y*camera.image.x+xy.x;
    let forward=unit(camera.look_at.xyz-camera.eye.xyz);let reference_up=select(vec3<f32>(0.,0.,1.),vec3<f32>(0.,1.,0.),abs(forward.z)>.99);let right=unit(cross(forward,reference_up));let up=cross(right,forward);
    let jitter=vec2<f32>(random_uniform(pixel,camera.image.z*2u+0x40000000u),random_uniform(pixel,camera.image.z*2u+0x40000001u));
    let uv=2.*(vec2<f32>(xy)+jitter)/vec2<f32>(camera.image.xy)-1.;
    let direction=unit(forward+camera.look_at.w*(uv.x*f32(camera.image.x)/f32(camera.image.y)*right-uv.y*up));
    var rgb=vec3<f32>(0.);
    // Each bin map stores integrated energy in that bin, so summation needs no
    // wavelength-width factor. Negative linear RGB is retained through the sum.
    for(var bin=0u;bin<WAVELENGTH_BINS;bin++){
        var response=v3(tables[40u]+bin*3u);if(bin==0u&&tables[42u]!=0u&&camera.options.y>.5){response=vec3<f32>(.22,.02,.9);}if(all(abs(response)<vec3<f32>(1e-9))){continue;}
        rgb+=response*spectral_camera(camera.eye.xyz,direction,bin,pixel);
    }
    var average=rgb;
    if(camera.image.z>0u){average=(accumulation[pixel].xyz*f32(camera.image.z)+rgb)/f32(camera.image.z+1u);}
    accumulation[pixel]=vec4<f32>(average,1.);
    let exposed=max(average*camera.eye.w,vec3<f32>(0.));
    // Monotone photographic compression; no screen-space glow/bloom deposition.
    let mapped=exposed/(1.+max(max(exposed.x,exposed.y),exposed.z));
    textureStore(camera_image,vec2<i32>(xy),vec4<f32>(srgb(mapped),1.));
}
