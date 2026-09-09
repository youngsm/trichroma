// Photon-density maps. Values are accumulated as f32 via u32 compare/exchange.
// Energy is relative to one450nm photon, before division by the launched N.
@group(0) @binding(5) var<storage,read_write> wall_map:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> volume_map:array<atomic<u32>>;
@group(0) @binding(7) var<storage,read_write> emission_map:array<atomic<u32>>;
@group(0) @binding(11) var<storage,read_write> fill_wall_map:array<atomic<u32>>;
const ROOM:vec3<f32>=vec3<f32>(300.,200.,120.);
const VOLUME_SHAPE:vec3<u32>=vec3<u32>(64u,40u,24u);
const VOLUME_CELLS:u32=61440u;
const WAVELENGTH_BINS:u32=64u;
fn energy_bin(wavelength:f32)->u32{
    var bin=u32(clamp((wavelength-280.)/7.1875,0.,63.));
    if(bin>0u&&wavelength<280.+f32(bin)*7.1875){bin--;}
    if(bin<63u&&wavelength>=280.+f32(bin+1u)*7.1875){bin++;}
    return bin;
}
fn add_wall(index:u32,value:f32){
    var old=atomicLoad(&wall_map[index]);
    loop{let next=bitcast<u32>(bitcast<f32>(old)+value);let result=atomicCompareExchangeWeak(&wall_map[index],old,next);if(result.exchanged){break;}old=result.old_value;}
}
fn add_volume(index:u32,value:f32){
    if(value==0.){return;}var old=atomicLoad(&volume_map[index]);
    loop{let next=bitcast<u32>(bitcast<f32>(old)+value);let result=atomicCompareExchangeWeak(&volume_map[index],old,next);if(result.exchanged){break;}old=result.old_value;}
}
fn add_emission(index:u32,value:f32){
    var old=atomicLoad(&emission_map[index]);
    loop{let next=bitcast<u32>(bitcast<f32>(old)+value);let result=atomicCompareExchangeWeak(&emission_map[index],old,next);if(result.exchanged){break;}old=result.old_value;}
}
fn add_fill_wall(index:u32,value:f32){
    var old=atomicLoad(&fill_wall_map[index]);
    loop{let next=bitcast<u32>(bitcast<f32>(old)+value);let result=atomicCompareExchangeWeak(&fill_wall_map[index],old,next);if(result.exchanged){break;}old=result.old_value;}
}
// Face order +/-X,+/-Y,+/-Z; UV coordinates use the other two world axes.
fn face_index(normal:vec3<f32>)->u32{
    let n=abs(normal);var axis=0u;if(n.y>n.x){axis=1u;}if(n.z>n[axis]){axis=2u;}
    return axis*2u+select(1u,0u,normal[axis]>=0.);
}
fn face_uv(position:vec3<f32>,half:vec3<f32>,face:u32)->vec2<f32>{
    var p=position.yz;var h=half.yz;
    if(face/2u==1u){p=position.xz;h=half.xz;}
    if(face/2u==2u){p=position.xy;h=half.xy;}
    return clamp(.5+.5*p/h,vec2<f32>(0.),vec2<f32>(.999999));
}
fn face_area(half:vec3<f32>,face:u32,resolution:u32)->f32{
    var a=half.y*half.z;if(face/2u==1u){a=half.x*half.z;}if(face/2u==2u){a=half.x*half.y;}
    return 4.*a/f32(resolution*resolution);
}
fn deposit_wall(position:vec3<f32>,normal:vec3<f32>,wavelength:f32,packet_scale:f32,is_fill:bool){
    if(is_fill){let face=face_index(normal);let uv=vec2<u32>(face_uv(position,ROOM,face)*32.);add_fill_wall(((face*32u+uv.y)*32u+uv.x)*WAVELENGTH_BINS+energy_bin(wavelength),450./wavelength*packet_scale);return;}
    let face=face_index(normal);let uv=vec2<u32>(face_uv(position,ROOM,face)*128.);
    add_wall(((face*128u+uv.y)*128u+uv.x)*WAVELENGTH_BINS+energy_bin(wavelength),450./wavelength*packet_scale);
}
fn deposit_scatter(position:vec3<f32>,polarization:vec3<f32>,wavelength:f32,packet_scale:f32){
    let cell=vec3<u32>(clamp((position+ROOM)/(2.*ROOM)*vec3<f32>(VOLUME_SHAPE),vec3<f32>(0.),vec3<f32>(VOLUME_SHAPE)-1.));
    let voxel=(cell.z*40u+cell.y)*64u+cell.x;let base=(voxel*WAVELENGTH_BINS+energy_bin(wavelength))*6u;
    let p=unit(polarization);let energy=450./wavelength*packet_scale;
    add_volume(base,energy*p.x*p.x);add_volume(base+1u,energy*p.y*p.y);add_volume(base+2u,energy*p.z*p.z);
    add_volume(base+3u,energy*p.x*p.y);add_volume(base+4u,energy*p.x*p.z);add_volume(base+5u,energy*p.y*p.z);
}
fn deposit_emission(position:vec3<f32>,normal:vec3<f32>,direction:vec3<f32>,wavelength:f32,packet_scale:f32,triangle:u32){
    if(tables[43u]!=0u){
        let chart=bitcast<i32>(tables[tables[43u]+triangle]);
        if(chart<0){return;}
        let hemisphere=select(1u,0u,dot(direction,normal)>=0.);
        add_emission((2u*u32(chart)+hemisphere)*WAVELENGTH_BINS+energy_bin(wavelength),450./wavelength*packet_scale);
        return;
    }
    let face=face_index(normal);let uv=vec2<u32>(face_uv(position-v3(tables[41u]+70u),v3(tables[41u]+73u),face)*64.);
    let hemisphere=select(1u,0u,dot(direction,normal)>=0.);
    add_emission(((((face*2u+hemisphere)*64u+uv.y)*64u+uv.x)*WAVELENGTH_BINS)+energy_bin(wavelength),450./wavelength*packet_scale);
}
