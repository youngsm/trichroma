// Bounded optical Monte Carlo for the three exported spectral-lab scenes.
// Philox streams, Fresnel sampling, polarized Rayleigh and surface emission
// follow the Chroma spectral kernels. Numerical operations use WGSL f32.
struct Configuration { counts: vec4<u32>, options: vec4<u32>, plot: vec4<f32>, }
@group(0) @binding(0) var<storage, read> tables: array<u32>;
@group(0) @binding(1) var<uniform> config: Configuration;
@group(0) @binding(2) var<storage, read_write> statistics: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> paths: array<u32>;
@group(0) @binding(4) var<storage, read_write> final_state: array<u32>;
const IN_SPECTRUM: u32 = 32u;
const OUT_SPECTRUM: u32 = 124u;
const TIME_HIST: u32 = 216u;
const DELAY_HIST: u32 = 344u;
const SCATTERED_HIST: u32 = 424u;
const SOURCE_SCATTER_HIST: u32 = 456u;
const DISPERSION_HIST: u32 = 488u;
const STAT_WORDS: u32 = 15788u; // 85 wavelength x 180 angle bins follow offset488.
const PI2: f32 = 6.283185307179586;
const END: u32 = 0xffffffffu;
fn f(i: u32) -> f32 { return bitcast<f32>(tables[i]); }
fn v3(i: u32) -> vec3<f32> { return vec3<f32>(f(i), f(i+1u), f(i+2u)); }
fn wide_multiply(a: u32, b: u32) -> vec2<u32> {
    let al = a & 65535u; let ah = a >> 16u;
    let bl = b & 65535u; let bh = b >> 16u;
    let first = al*bl;
    let middle = ah*bl+(first >> 16u);
    let upper = al*bh+(middle & 65535u);
    return vec2<u32>((upper << 16u)|(first & 65535u), ah*bh+(middle >> 16u)+(upper >> 16u));
}
fn philox_counter(counter: vec4<u32>, key: vec2<u32>) -> u32 {
    var c = counter;
    var k = key;
    for (var round = 0u; round < 10u; round++) {
        let a = wide_multiply(c.x, 0xd2511f53u);
        let b = wide_multiply(c.z, 0xcd9e8d57u);
        c = vec4<u32>(b.y ^ c.y ^ k.x, b.x, a.y ^ c.w ^ k.y, a.x);
        k += vec2<u32>(0x9e3779b9u, 0xbb67ae85u);
    }
    return c.x;
}
fn random_uniform(id: u32, stream: u32) -> f32 {
    return (f32(philox_counter(vec4<u32>(id,0u,stream,0u),config.counts.yz) >> 9u)+.5)*1.1920928955078125e-7;
}
fn interpolate(property: u32, row: u32, wavelength: f32) -> f32 {
    let fraction = clamp((wavelength-f(5u))/f(6u), 0., f32(tables[4u]-1u));
    let lo = min(u32(fraction), tables[4u]-2u); let alpha = fraction-f32(lo);
    let start = tables[property]+row*tables[4u];
    let left = f(start+lo); let right = f(start+lo+1u);
    if (alpha <= 0. || left == right) { return left; }
    if (alpha >= 1.) { return right; }
    return (1.-alpha)*left+alpha*right;
}
fn sample_wavelength(row: u32, u: f32) -> f32 {
    let start = tables[25u]+row*tables[4u];
    var lo = 0u; var hi = tables[4u]-1u;
    loop {
        if (hi-lo <= 1u) { break; }
        let middle = (lo+hi)/2u;
        if (f(start+middle) <= u) { lo = middle; } else { hi = middle; }
    }
    let alpha = (u-f(start+lo))/max(f(start+hi)-f(start+lo), 1e-30);
    return f(5u)+(f32(lo)+alpha)*f(6u);
}
fn sample_delay(row: u32, u: f32) -> f32 {
    var lo = tables[tables[26u]+row]; var hi = tables[tables[26u]+row+1u]-1u;
    loop {
        if (hi-lo <= 1u) { break; }
        let middle = (lo+hi)/2u;
        if (f(tables[28u]+middle) <= u) { lo = middle; } else { hi = middle; }
    }
    let a = f(tables[28u]+lo); let b = f(tables[28u]+hi);
    var fraction = (u-a)/max(b-a, 1e-30);
    let p0 = f(tables[29u]+lo); let p1 = f(tables[29u]+hi);
    if (p0 >= 0.) {
        let root = sqrt(max(0., p0*p0+fraction*(p1*p1-p0*p0)));
        fraction = fraction*(p0+p1)/max(p0+root, 1e-30);
    }
    return f(tables[27u]+lo)+fraction*(f(tables[27u]+hi)-f(tables[27u]+lo));
}
// WGSL native transcendental accuracy is implementation-dependent. In particular,
// SwiftShader sin/cos can be ~2e-4 away from unit norm. Restrict the polynomial
// interval explicitly so source and scattering angles retain float32 accuracy.
fn sine_cosine(angle: f32) -> vec2<f32> {
    var x = angle-floor((angle+3.141592653589793)/PI2)*PI2;
    var cosine_sign = 1.;
    if (x > 1.5707963267948966) { x=3.141592653589793-x; cosine_sign=-1.; }
    if (x < -1.5707963267948966) { x=-3.141592653589793-x; cosine_sign=-1.; }
    let x2=x*x;
    let sine=x*(1.+x2*(-0.16666666666666666+x2*(0.008333333333333333+x2*(-0.0001984126984126984+x2*(0.0000027557319223985893+x2*(-0.00000002505210838544172))))));
    let cosine=1.+x2*(-0.5+x2*(0.041666666666666664+x2*(-0.001388888888888889+x2*(0.0000248015873015873+x2*(-0.0000002755731922398589)))));
    return vec2<f32>(sine,cosine_sign*cosine);
}
fn unit(v: vec3<f32>) -> vec3<f32> { return v/sqrt(max(dot(v,v),1e-30)); }
fn tangent(n: vec3<f32>) -> vec3<f32> {
    if (abs(n.z) > .9) { return unit(vec3<f32>(n.z, 0., -n.x)); }
    return unit(vec3<f32>(-n.y, n.x, 0.));
}
fn hemisphere(n: vec3<f32>, u: f32, phi_u: f32, lambert: bool) -> vec3<f32> {
    let t = tangent(n); let q = cross(n, t);
    let cosine = select(u, sqrt(u), lambert); let sine = sqrt(max(0., 1.-cosine*cosine));
    let angle=sine_cosine(PI2*phi_u);
    return unit(cosine*n+sine*angle.y*t+sine*angle.x*q);
}
fn random_polarization(direction: vec3<f32>, u: f32) -> vec3<f32> {
    let t = tangent(direction);
    let angle=sine_cosine(PI2*u);
    return unit(angle.y*t+angle.x*cross(direction,t));
}
struct OpticalRay { direction: vec3<f32>, polarization: vec3<f32>, reflected: bool, }
fn rayleigh(polarization: vec3<f32>, id: u32, stream: u32) -> OpticalRay {
    let p = unit(polarization);
    var b = vec3<f32>(-p.y, p.x, 0.);
    if (abs(p.z) >= .9) { b = vec3<f32>(0., -p.z, p.y); }
    b = unit(b); let q = cross(p, b);
    let a = random_uniform(id, stream+2u); let c = random_uniform(id, stream+3u); let d = random_uniform(id, stream+4u);
    let cosine = 2.*max(min(a,c), min(max(a,c),d))-1.;
    let sine = sqrt(max(0., 1.-cosine*cosine)); let phi = PI2*random_uniform(id, stream+5u);
    let angle=sine_cosine(phi); let t=angle.y*b+angle.x*q;
    return OpticalRay(unit(cosine*p+sine*t),unit(sine*p-cosine*t),false);
}
fn fresnel(direction: vec3<f32>, polarization: vec3<f32>, normal: vec3<f32>, n1: f32, n2: f32, id: u32, stream: u32) -> OpticalRay {
    let ci = clamp(-dot(direction,normal), 0., 1.); let eta = n1/n2;
    let st2 = eta*eta*max(0., 1.-ci*ci); let tir = st2 > 1.; let ct = sqrt(max(0.,1.-st2));
    let ds = n1*ci+n2*ct; let dp = n2*ci+n1*ct;
    let amplitude_s = select((n1*ci-n2*ct)/max(abs(ds),1e-20), 0., abs(ds)<=1e-20);
    let amplitude_p = select((n2*ci-n1*ct)/max(abs(dp),1e-20), 0., abs(dp)<=1e-20);
    let raw_s = cross(direction,normal);
    var s = polarization;
    if (dot(raw_s,raw_s) >= 1e-12) { s = unit(raw_s); }
    let coefficient = dot(polarization,s);
    let choose_s = random_uniform(id,stream+17u) < clamp(coefficient*coefficient,0.,1.);
    let reflectance = select(amplitude_p*amplitude_p,amplitude_s*amplitude_s,choose_s);
    let reflected = tir || random_uniform(id,stream+18u) < reflectance;
    var out_direction = eta*direction+(eta*ci-ct)*normal;
    if (reflected) { out_direction = direction-2.*dot(direction,normal)*normal; }
    var out_polarization = s;
    if (!choose_s) { out_polarization = unit(cross(s,out_direction)); }
    return OpticalRay(unit(out_direction),unit(out_polarization),reflected);
}
struct Boundary { triangle: u32, distance: f32, }
fn nearest_boundary(origin: vec3<f32>, direction: vec3<f32>, previous: u32) -> Boundary {
    // CAMERA_BVH_HOOK
    var best = Boundary(END, 1e30);
    for (var tri = 0u; tri < tables[1u]; tri++) {
        if (tri == previous) { continue; }
        let base = tables[8u]+tri*9u; let a = v3(base);
        let e1 = v3(base+3u)-a; let e2 = v3(base+6u)-a;
        let h = cross(direction,e2); let determinant = dot(e1,h);
        if (abs(determinant) <= 1.1920928955078125e-7) { continue; }
        let reciprocal = 1./determinant; let s = origin-a;
        let u = reciprocal*dot(s,h); let q = cross(s,e1);
        let v = reciprocal*dot(direction,q); let distance = reciprocal*dot(e2,q);
        if (u>=-1e-6 && u<=1.000001 && v>=-1e-6 && u+v<=1.000001 && distance>1e-6 && distance<best.distance) {
            best = Boundary(tri,distance);
        }
    }
    return best;
}
fn next_float(value: f32, positive: bool) -> f32 {
    if (value == 0.) { return select(-1.1754943508222875e-38,1.1754943508222875e-38,positive); }
    var word = bitcast<u32>(value);
    if ((value > 0.) == positive) { word++; } else { word--; }
    return bitcast<f32>(word);
}
fn boundary_origin(position: vec3<f32>, direction: vec3<f32>, tri: u32) -> vec3<f32> {
    let base = tables[8u]+tri*9u; let a = v3(base);
    let u = v3(base+3u)-a; let v = v3(base+6u)-a; let n = cross(u,v);
    let extent3 = abs(u)+abs(v)+abs(abs(u)-abs(v));
    let extent = max(max(extent3.x,extent3.y),extent3.z);
    let error = 1.1920928955078125e-7*abs(a)+vec3<f32>(3.5762786865234375e-7*extent);
    let clearance = dot(abs(n),error); let residual = dot(position-a,n);
    let side = select(-1.,1.,dot(direction,n)>=0.);
    var shifted = position+((side*clearance-residual)/max(dot(n,n),1e-30))*n;
    for (var axis=0u;axis<3u;axis++) { if(n[axis]!=0.) { shifted[axis]=next_float(shifted[axis],side*n[axis]>0.); } }
    return shifted;
}
fn histogram(offset: u32, value: f32, low: f32, high: f32, count: u32) {
    // All laboratory bin widths are exactly representable. Correct a rounded
    // quotient at a boundary so the GPU agrees with independent edge searches.
    let width=(high-low)/f32(count);
    var bin=u32(clamp((value-low)/width,0.,f32(count-1u)));
    if (bin>0u && value<low+f32(bin)*width) { bin--; }
    if (bin+1u<count && value>=low+f32(bin+1u)*width) { bin++; }
    atomicAdd(&statistics[offset+bin],1u);
}
fn path_vertex(id: u32, vertex: u32, position: vec3<f32>, time: f32, wavelength: f32, flags: u32, flight: f32, arrival_time: f32) {
    if (id >= config.counts.w) { return; }
    let base = (id*(config.options.x+1u)+vertex)*8u;
    paths[base] = bitcast<u32>(position.x); paths[base+1u] = bitcast<u32>(position.y); paths[base+2u] = bitcast<u32>(position.z);
    paths[base+3u] = bitcast<u32>(time); paths[base+4u] = bitcast<u32>(wavelength); paths[base+5u] = flags;
    paths[base+6u] = bitcast<u32>(flight); paths[base+7u] = bitcast<u32>(arrival_time);
    atomicStore(&statistics[STAT_WORDS+id],vertex+1u);
}
@compute @workgroup_size(128)
fn simulate(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let id = invocation.x+invocation.y*8388480u; if (id >= config.counts.x) { return; }
    var position = v3(31u);
    position.y += (random_uniform(id,0x10000000u)-.5)*f(36u);
    position.z += (random_uniform(id,0x10000001u)-.5)*f(36u);
    var initial_position = position;
    var direction = vec3<f32>(1.,0.,0.);
    let angle = PI2*random_uniform(id,0x10000002u);
    let source_angle=sine_cosine(angle);
    var polarization=unit(vec3<f32>(0.,source_angle.y,source_angle.x));
    if(config.options.w==1u){polarization=vec3<f32>(0.,1.,0.);}
    if(config.options.w==2u){polarization=vec3<f32>(0.,0.,1.);}
    var wavelength = f(34u)+(f(35u)-f(34u))*random_uniform(id,0x10000003u);
    // CAMERA_SOURCE_HOOK
    let source_wavelength = wavelength;
    var time = 0.; var flight = 0.; var delay_total = 0.; var flags = 0u;
    var previous = END; var channel = -1; var steps = 0u;
    path_vertex(id,0u,position,time,wavelength,flags,flight,time);
    histogram(IN_SPECTRUM,wavelength,280.,740.,92u);
    histogram(SOURCE_SCATTER_HIST,wavelength,390.,710.,32u);
    for(var step=0u;step<config.options.x;step++) {
        steps = step+1u; let stream = step*32u;
        let hit = nearest_boundary(position,direction,previous);
        if(hit.triangle==END){flags|=1u;
            // CAMERA_ESCAPE_HOOK
            break;
        }
        let tri=hit.triangle; let normal=v3(tables[9u]+tri*3u);
        let outward=dot(direction,normal)>0.;
        let m1=tables[tables[10u]+tri]; let m2=tables[tables[11u]+tri];
        let incident=select(m2,m1,outward); let other=select(m1,m2,outward);
        let inward=select(normal,-normal,outward);
        let absorption=interpolate(15u,incident,wavelength); let scattering=interpolate(16u,incident,wavelength);
        var da=1e30;var ds=1e30;
        if(absorption<1e30){da=-absorption*log(random_uniform(id,stream));}
        if(scattering<1e30){ds=-scattering*log(random_uniform(id,stream+1u));}
        var bulk_absorbed=da<=ds && da<=hit.distance;
        var scattered=ds<da && ds<=hit.distance;
        var travel=min(hit.distance,min(da,ds));
        // Preserve an already selected bulk collision. If f32 reconstruction
        // lands on/outside the upcoming interface, move its origin into the
        // incident material by the same conservative bound used at interfaces.
        // This retains the sampled scattering direction and its random stream.
        position+=travel*direction;flight+=travel;
        if(scattered){
            let boundary_point=v3(tables[8u]+tri*9u);
            if(dot(position-boundary_point,inward)<=0.){
                position=boundary_origin(position,inward,tri);
                atomicAdd(&statistics[24u],1u);
            }
        }
        time+=travel/max(interpolate(17u,incident,wavelength),1e-30);
        let arrival_time=time;
        if(bulk_absorbed){flags|=2u;previous=END;}
        else if(scattered){
            // CAMERA_SCATTER_HOOK
            let ray=rayleigh(polarization,id,stream);direction=ray.direction;polarization=ray.polarization;flags|=16u;previous=END;
        }else{
            previous=tri;
            let signed_surface=bitcast<i32>(tables[tables[12u]+tri]);
            var handled=false;
            if(signed_surface>=0 && tables[tables[18u]+u32(signed_surface)]!=0u){
                let surface=u32(signed_surface);let model=tables[tables[19u]+surface];
                let absorb=interpolate(21u,surface,wavelength);
                var detect=0.;if(model==0u){detect=interpolate(20u,surface,wavelength);}
                let diffuse=interpolate(22u,surface,wavelength);let specular=interpolate(23u,surface,wavelength);
                let draw=random_uniform(id,stream+6u);
                if(draw<absorb){
                    handled=true;
                    if(model==2u && random_uniform(id,stream+7u)<interpolate(24u,surface,wavelength)){
                        wavelength=sample_wavelength(surface,random_uniform(id,stream+8u));
                        let delay=sample_delay(surface,random_uniform(id,stream+9u));time+=delay;delay_total+=delay;
                        let side=select(1.,-1.,random_uniform(id,stream+10u)<f(tables[30u]+surface));
                        direction=hemisphere(side*normal,random_uniform(id,stream+11u),random_uniform(id,stream+12u),false);
                        polarization=random_polarization(direction,random_uniform(id,stream+13u));flags|=128u;
                        // CAMERA_EMISSION_HOOK
                    }else{flags|=8u;}
                }else if(draw<absorb+detect){handled=true;flags|=4u;channel=bitcast<i32>(tables[tables[13u]+tri]);}
                else if(draw<absorb+detect+diffuse){
                    handled=true;direction=hemisphere(inward,random_uniform(id,stream+14u),random_uniform(id,stream+15u),true);
                    polarization=random_polarization(direction,random_uniform(id,stream+16u));flags|=32u;
                }else if(draw<absorb+detect+diffuse+specular){
                    handled=true;direction-=2.*dot(direction,inward)*inward;polarization-=2.*dot(polarization,inward)*inward;flags|=64u;
                }
            }
            if(!handled){
                let ray=fresnel(direction,polarization,inward,interpolate(14u,incident,wavelength),interpolate(14u,other,wavelength),id,stream);
                direction=ray.direction;polarization=ray.polarization;flags|=select(256u,64u,ray.reflected);
            }
            if((flags&15u)==0u){position=boundary_origin(position,direction,tri);}
        }
        // CAMERA_BOUNDARY_HOOK
        path_vertex(id,steps,position,time,wavelength,flags,flight,arrival_time);
        if((flags&15u)!=0u){break;}
    }
    if((flags&15u)==0u){flags|=0x40000000u;atomicAdd(&statistics[5u],1u);
        // CAMERA_TAIL_HOOK
    }
    if(!all(abs(position)<vec3<f32>(3.4e38)) || !all(abs(direction)<vec3<f32>(3.4e38)) || !(abs(time)<3.4e38)){
        flags|=0x80000000u;atomicAdd(&statistics[6u],1u);
    }
    atomicAdd(&statistics[0u],1u);atomicMax(&statistics[12u],steps);atomicMax(&statistics[16u],bitcast<u32>(time));
    if((flags&4u)!=0u){
        atomicAdd(&statistics[1u],1u);histogram(OUT_SPECTRUM,wavelength,280.,740.,92u);
        histogram(TIME_HIST,time,0.,config.plot.x,128u);if(time>=config.plot.x){atomicAdd(&statistics[14u],1u);}
        if(config.options.y==0u && position.x>299.){
            atomicAdd(&statistics[13u],1u);
            let wb=u32(clamp((wavelength-380.)/4.,0.,84.));
            let angle=atan2(direction.y,direction.x)*57.29577951308232;
            let ab=u32(clamp(angle+90.,0.,179.));atomicAdd(&statistics[DISPERSION_HIST+ab*85u+wb],1u);
        }
    }
    if((flags&8u)!=0u){atomicAdd(&statistics[2u],1u);}
    if((flags&2u)!=0u){atomicAdd(&statistics[3u],1u);}
    if((flags&1u)!=0u){atomicAdd(&statistics[4u],1u);}
    if((flags&128u)!=0u){atomicAdd(&statistics[7u],1u);histogram(DELAY_HIST,delay_total,0.,200.,80u);}
    if((flags&16u)!=0u){atomicAdd(&statistics[8u],1u);histogram(SCATTERED_HIST,source_wavelength,390.,710.,32u);}
    if((flags&64u)!=0u){atomicAdd(&statistics[9u],1u);}
    if((flags&256u)!=0u){atomicAdd(&statistics[10u],1u);}
    if(id<config.options.z){
        let base=id*20u;
        final_state[base]=bitcast<u32>(position.x);final_state[base+1u]=bitcast<u32>(position.y);final_state[base+2u]=bitcast<u32>(position.z);final_state[base+3u]=bitcast<u32>(time);
        final_state[base+4u]=bitcast<u32>(direction.x);final_state[base+5u]=bitcast<u32>(direction.y);final_state[base+6u]=bitcast<u32>(direction.z);final_state[base+7u]=bitcast<u32>(wavelength);
        final_state[base+8u]=bitcast<u32>(polarization.x);final_state[base+9u]=bitcast<u32>(polarization.y);final_state[base+10u]=bitcast<u32>(polarization.z);final_state[base+11u]=bitcast<u32>(source_wavelength);
        final_state[base+12u]=flags;final_state[base+13u]=bitcast<u32>(channel);final_state[base+14u]=previous;final_state[base+15u]=steps;
        final_state[base+16u]=bitcast<u32>(delay_total);final_state[base+17u]=bitcast<u32>(flight);final_state[base+18u]=bitcast<u32>(initial_position.y);final_state[base+19u]=bitcast<u32>(initial_position.z);
    }
}
// Independent counter/key vectors exercise the same RNG as full transport.
@compute @workgroup_size(64)
fn rng_probe(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let id=invocation.x;if(id>=config.counts.x){return;}
    let b=id*5u;
    let bits=philox_counter(vec4<u32>(tables[b],tables[b+1u],tables[b+4u],0u),vec2<u32>(tables[b+2u],tables[b+3u]));
    final_state[id]=bitcast<u32>((f32(bits>>9u)+.5)*1.1920928955078125e-7);
}
