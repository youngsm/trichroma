// Appended to the unchanged shared detector traversal shader. Units: mm, ns, nm.
// settings.a[4]: PMT count, pulse decay (ns), accumulated-hit weight, display gain.
struct Settings { a: array<vec4<f32>, 5> }
struct SourceStep { start: vec4<f32>, end: vec4<f32>, direction: vec4<f32> }
struct Flight { start: vec4<f32>, end: vec4<f32>, polarization: vec4<f32>, identity: vec4<f32> }
@group(0) @binding(5) var depth_output: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var<uniform> settings: Settings;
@group(0) @binding(7) var<storage, read> sources: array<SourceStep>;
@group(0) @binding(8) var<storage, read_write> flights: array<Flight>;
@group(0) @binding(9) var<storage, read_write> flight_counts: array<u32>;
@group(0) @binding(10) var<storage, read_write> statistics: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> debug_source: array<vec4<f32>>;
// One fixed slot per transported photon. Zero sensor ID means no PMT arrival.
@group(0) @binding(12) var<storage, read_write> pmt_hits: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read> hit_offsets: array<u32>;
@group(0) @binding(14) var<storage, read> hit_times: array<f32>;
@group(0) @binding(15) var<storage, read_write> hit_states: array<vec4<f32>>;
@group(0) @binding(16) var sensor_output: texture_storage_2d<r32uint, write>;
const C_MM_NS: f32 = 299.792458;
const PI_EVENT: f32 = 3.141592653589793;
const MAX_FLIGHTS: u32 = 32u;
fn random_uniform(r:ptr<function,u32>)->f32 {
    *r=*r*747796405u+2891336453u;
    let word=((*r>>((*r>>28u)+4u))^*r)*277803737u;
    // Float32 rounding must not turn the largest draw into exactly one.
    return min(.9999999403953552,(f32(((word>>22u)^word)>>8u)+.5)/16777216.);
}
fn water_index(w:f32)->f32{return 1.322+3000./(w*w);}
fn group_index(w:f32)->f32{return 1.322+9000./(w*w);}
fn water_absorption(w:f32)->f32 {
    let wavelength=array<f32,8>(360.,400.,450.,500.,550.,600.,650.,700.);
    let meters=array<f32,8>(20.,50.,35.,15.,7.,2.5,1.,.6);
    for(var i=0u;i<7u;i++){
        if(w<=wavelength[i+1u]){return 1000.*exp(mix(log(meters[i]),log(meters[i+1u]),(w-wavelength[i])/(wavelength[i+1u]-wavelength[i])));}
    }
    return 600.;
}
fn random_sphere(r:ptr<function,u32>)->vec3<f32>{
    let z=2.*random_uniform(r)-1.;let phi=2.*PI_EVENT*random_uniform(r);
    let s=sqrt(max(0.,1.-z*z));return vec3<f32>(s*cos(phi),s*sin(phi),z);
}
@compute @workgroup_size(128)
fn propagate(@builtin(global_invocation_id) gid:vec3<u32>){
    let id=gid.x+u32(settings.a[0].y);if(id>=u32(settings.a[0].x)){return;}
    var rng=((id+1u)*2654435761u)^u32(settings.a[0].w);
    let pick=random_uniform(&rng);var lo=0u;var hi=u32(settings.a[0].z)-1u;
    while(lo<hi){let mid=(lo+hi)/2u;if(pick<sources[mid].direction.w){hi=mid;}else{lo=mid+1u;}}
    let step=sources[lo];let beta=step.end.w;let axis=normalize(step.direction.xyz);
    atomicAdd(&statistics[0],1u);atomicAdd(&statistics[128u+lo],1u);
    let maximum=max(0.,1.-1./pow(beta*water_index(360.),2.));
    var wavelength=0.;
    for(var attempt=0u;attempt<256u;attempt++){
        let w=1./mix(1./360.,1./700.,random_uniform(&rng));
        if(random_uniform(&rng)*maximum<max(0.,1.-1./pow(beta*water_index(w),2.))){wavelength=w;break;}
    }
    if(wavelength==0.){atomicAdd(&statistics[7],1u);return;}
    atomicAdd(&statistics[16u+min(67u,u32((wavelength-360.)/5.))],1u);
    let fraction=random_uniform(&rng);var position=mix(step.start.xyz,step.end.xyz,fraction);
    var time=step.start.w+fraction*length(step.end.xyz-step.start.xyz)/(beta*C_MM_NS);
    let cosine=1./(beta*water_index(wavelength));let sine=sqrt(max(0.,1.-cosine*cosine));
    let phi=2.*PI_EVENT*random_uniform(&rng);
    let helper=select(vec3<f32>(0.,1.,0.),vec3<f32>(1.,0.,0.),abs(axis.y)>.9);
    let u=normalize(cross(helper,axis));let v=cross(axis,u);
    var direction=normalize(cosine*axis+sine*(cos(phi)*u+sin(phi)*v));
    var polarization=normalize(axis-cosine*direction);
    let stride=u32(settings.a[1].y);let slot=id/stride;
    let retained=id%stride==0u && slot<u32(settings.a[1].x);
    if(retained){atomicAdd(&statistics[8],1u);}
    let debug=id<u32(settings.a[3].z);
    if(debug){debug_source[id*4u]=vec4<f32>(position,time);debug_source[id*4u+1u]=vec4<f32>(direction,wavelength);debug_source[id*4u+2u]=vec4<f32>(polarization,beta);debug_source[id*4u+3u]=vec4<f32>(f32(lo),0.,0.,0.);}
    let optical_off=settings.a[3].w>.5;
    let labs=select(water_absorption(wavelength),1e30,optical_off);
    let lscatter=select(80000.*pow(wavelength/450.,4.),1e30,optical_off);
    let ng=group_index(wavelength);
    for(var bounce=0u;bounce<MAX_FLIGHTS;bounce++){
        let boundary=trace(position,direction);
        if(boundary.triangle==END){atomicAdd(&statistics[4],1u);return;}
        let absorb=-labs*log(random_uniform(&rng));let scatter=-lscatter*log(random_uniform(&rng));
        let distance=min(boundary.distance,min(absorb,scatter));
        let end=position+distance*direction;let arrival=time+distance*ng/C_MM_NS;
        var identity=vec4<f32>(f32(id),-2.,-1.,-1.);
        let hit_boundary=boundary.distance<=absorb && boundary.distance<=scatter;
        let absorbed=!hit_boundary && absorb<=scatter;
        if(hit_boundary){identity=vec4<f32>(f32(id),f32(boundary.group),f32(boundary.instance),f32(boundary.triangle));}
        if(absorbed){identity.y=-3.;}
        if(retained){flights[slot*MAX_FLIGHTS+bounce]=Flight(vec4<f32>(position,time),vec4<f32>(end,arrival),vec4<f32>(polarization,wavelength),identity);flight_counts[slot]=bounce+1u;}
        atomicAdd(&statistics[9],1u);
        if(debug){debug_source[id*4u+3u]=vec4<f32>(f32(lo),arrival,f32(bounce+1u),identity.y);}
        if(hit_boundary && boundary.group==1u){pmt_hits[id]=vec4<f32>(arrival,f32(boundary.instance+1u),wavelength,f32(id));}
        if(hit_boundary){atomicAdd(&statistics[select(2u,1u,boundary.group==1u)],1u);return;}
        if(absorbed){atomicAdd(&statistics[3],1u);return;}
        atomicAdd(&statistics[6],1u);
        var next=vec3<f32>(0.);
        for(var attempt=0u;attempt<256u;attempt++){
            let trial=random_sphere(&rng);
            if(random_uniform(&rng)<1.-pow(dot(polarization,trial),2.)){next=trial;break;}
        }
        if(dot(next,next)<.5){atomicAdd(&statistics[7],1u);return;}
        polarization=normalize(polarization-dot(polarization,next)*next);
        position=end;time=arrival;direction=next;
    }
    atomicAdd(&statistics[5],1u);
}
@compute @workgroup_size(8,8)
fn detector_camera(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_id) lane:vec3<u32>){
    let columns=(cfg.dimensions.x+7u)/8u;let block=group.x+cfg.options.z;
    let pixel=vec2<u32>(block%columns,block/columns)*8u+lane.xy;
    if(any(pixel>=cfg.dimensions.xy)){return;}
    let uv=(vec2<f32>(pixel)+.5)/vec2<f32>(cfg.dimensions.xy);
    let direction=normalize(cfg.forward.xyz+cfg.right.xyz*((2.*uv.x-1.)*f32(cfg.dimensions.x)/f32(cfg.dimensions.y)*cfg.forward.w)+cfg.up.xyz*((1.-2.*uv.y)*cfg.forward.w));
    let hit=trace(cfg.eye.xyz,direction);
    var color=vec3<f32>(.018,.022,.03);
    if(hit.triangle!=END && hit.group==1u){color=.5*(hit.normal+vec3<f32>(1.));}
    textureStore(output,vec2<i32>(pixel),vec4<f32>(color,1.));
    textureStore(sensor_output,vec2<i32>(pixel),vec4<u32>(select(0u,hit.instance+1u,hit.triangle!=END && hit.group==1u),0u,0u,0u));
    textureStore(depth_output,vec2<i32>(pixel),vec4<f32>(hit.distance,0.,0.,0.));
}

// Sorted arrival lists let time scrubbing update all PMTs without transport or readback.
@compute @workgroup_size(128)
fn update_hits(@builtin(global_invocation_id) id:vec3<u32>){
    let sensor=id.x;if(sensor>=u32(settings.a[4].x)){return;}
    let first=hit_offsets[sensor];let end=hit_offsets[sensor+1u];
    var lo=first;var hi=end;
    let time=select(settings.a[2].x,1e30,settings.a[1].w>.5);
    while(lo<hi){let mid=(lo+hi)/2u;if(hit_times[mid]<=time){lo=mid+1u;}else{hi=mid;}}
    var latest=-1e30;if(lo>first){latest=hit_times[lo-1u];}
    // Every arrival contributes one pulse. Recompute from recorded times so
    // scrubbing, playback speed, and frame rate cannot change the signal.
    var amplitude=0.;
    if(settings.a[1].w<.5){
        let tau=max(settings.a[4].y,.001);
        for(var hit=first;hit<lo;hit++){
            amplitude+=exp(-max(0.,time-hit_times[hit])/tau);
        }
    }
    hit_states[sensor]=vec4<f32>(f32(lo-first),latest,amplitude,0.);
}

// RASTER MODULE
// Read-only aliases let the vertex shader consume actual compute-written flights.
struct Config {eye:vec4<f32>,forward:vec4<f32>,right:vec4<f32>,up:vec4<f32>,dimensions:vec4<u32>,options:vec4<u32>}
struct Settings {a:array<vec4<f32>,5>}
struct Flight {start:vec4<f32>,end:vec4<f32>,polarization:vec4<f32>,identity:vec4<f32>}
@group(0) @binding(0) var<storage,read> flights:array<Flight>;
@group(0) @binding(1) var<uniform> cfg:Config;
@group(0) @binding(2) var<uniform> settings:Settings;
@group(0) @binding(3) var detector_depth:texture_2d<f32>;
@group(0) @binding(4) var detector_color:texture_2d<f32>;
@group(0) @binding(5) var<storage,read> flight_counts:array<u32>;
@group(0) @binding(6) var detector_sensor:texture_2d<u32>;
@group(0) @binding(7) var<storage,read> hit_states:array<vec4<f32>>;
struct Vertex {@builtin(position) clip:vec4<f32>,@location(0) world:vec3<f32>,@location(1) color:vec3<f32>,@location(2) edge:f32,@location(3) age:f32}
fn camera_position(world:vec3<f32>)->vec3<f32>{let v=world-cfg.eye.xyz;return vec3<f32>(dot(v,cfg.right.xyz),dot(v,cfg.up.xyz),dot(v,cfg.forward.xyz));}
fn project(v:vec3<f32>)->vec4<f32>{return vec4<f32>(v.x/(cfg.forward.w*f32(cfg.dimensions.x)/f32(cfg.dimensions.y)),v.y/cfg.forward.w,.5*v.z,v.z);}
@vertex
fn photon_vertex(@builtin(vertex_index) index:u32,@builtin(instance_index) instance:u32)->Vertex{
    var out:Vertex;out.clip=vec4<f32>(2.,2.,.5,1.);out.world=vec3<f32>(0.);out.color=vec3<f32>(0.);out.edge=0.;out.age=0.;
    let slot=instance/32u;let step=instance%32u;
    if(step>=flight_counts[slot]){return out;}
    let flight=flights[instance];let dt=flight.end.w-flight.start.w;
    if(dt<=0.){return out;}
    var low=0.;var high=1.;
    if(settings.a[1].w<.5){
        low=clamp((settings.a[2].x-settings.a[2].y-flight.start.w)/dt,0.,1.);
        high=clamp((settings.a[2].x-flight.start.w)/dt,0.,1.);
    }
    if(high<=low){return out;}
    var a=mix(flight.start.xyz,flight.end.xyz,low);var b=mix(flight.start.xyz,flight.end.xyz,high);
    var ca=camera_position(a);var cb=camera_position(b);
    if(ca.z<=10. && cb.z<=10.){return out;}
    if(ca.z<10.){a=mix(a,b,(10.-ca.z)/(cb.z-ca.z));ca=camera_position(a);}
    if(cb.z<10.){b=mix(b,a,(10.-cb.z)/(ca.z-cb.z));cb=camera_position(b);}
    let pa=project(ca);let pb=project(cb);
    let delta=(pb.xy/pb.w-pa.xy/pa.w)*vec2<f32>(cfg.dimensions.xy);
    let side=vec2<f32>(-delta.y,delta.x)/max(length(delta),1e-6);
    let corner=array<vec2<f32>,6>(vec2<f32>(0.,-1.),vec2<f32>(1.,-1.),vec2<f32>(0.,1.),vec2<f32>(0.,1.),vec2<f32>(1.,-1.),vec2<f32>(1.,1.))[index];
    out.clip=mix(pa,pb,corner.x);
    out.clip=vec4<f32>(out.clip.xy+side*corner.y*1.4/vec2<f32>(cfg.dimensions.xy)*out.clip.w,out.clip.zw);
    let birth=flights[slot*32u].start.w;
    let point_time=mix(flight.start.w,flight.end.w,mix(low,high,corner.x));
    out.age=max(0.,select(settings.a[2].x,point_time,settings.a[1].w>.5)-birth);
    out.world=mix(a,b,corner.x);out.color=abs(flight.polarization.xyz);out.edge=corner.y;
    return out;
}
@fragment fn photon_fragment(in:Vertex)->@location(0) vec4<f32>{
    let location=vec2<i32>(in.clip.xy);
    let depth=textureLoad(detector_depth,location,0).x;
    if(length(in.world-cfg.eye.xyz)>depth+2.){discard;}
    let coverage=1.-smoothstep(.5,1.,abs(in.edge));
    return vec4<f32>(in.color,settings.a[2].z*coverage*select(1.,exp2(-in.age/max(settings.a[3].y,1e-6)),settings.a[3].y>0.));
}
@vertex fn fullscreen(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32>{
    let p=array<vec2<f32>,3>(vec2<f32>(-1.,-1.),vec2<f32>(3.,-1.),vec2<f32>(-1.,3.));return vec4<f32>(p[i],0.,1.);
}
@fragment fn background(@builtin(position) p:vec4<f32>)->@location(0) vec4<f32>{
    var color=textureLoad(detector_color,vec2<i32>(p.xy),0);
    let sensor=textureLoad(detector_sensor,vec2<i32>(p.xy),0).x;
    if(sensor>0u){
        var brightness=settings.a[2].w;
        if(settings.a[3].x>.5){
            let state=hit_states[sensor-1u];
            // Unit-height exponential pulses add linearly before display mapping.
            // A separate cumulative-count contribution keeps the ring visible.
            let signal=select(state.z+settings.a[4].z*state.x,state.x,settings.a[1].w>.5);
            let intensity=1.-exp(-max(0.,settings.a[4].w*signal));
            brightness=mix(brightness,1.,intensity);
        }
        // Preserve the normal RGB direction: arrival changes brightness only.
        color=vec4<f32>(color.xyz*brightness,1.);
    }
    return color;
}
