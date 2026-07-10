// solar-system — raymarched planet with plasma jet, magnetic field rings,
// orbiting probes, and a procedural cube-mapped starfield.
// Shadertoy-compatible: uses mainImage(out vec4, in vec2), iTime, iResolution, iMouse.

#define STEPS 100
#define MAX_DIST 100.
#define SURF_DIST .001

float hash2D(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}


float noise2D(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(mix(hash2D(i + vec2(0.0, 0.0)), hash2D(i + vec2(1.0, 0.0)), u.x),
               mix(hash2D(i + vec2(0.0, 1.0)), hash2D(i + vec2(1.0, 1.0)), u.x), u.y);
}

float fbm2D(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;

    for (int i = 0; i < 3; i++) {
        value += amplitude * noise2D(p);
        p *= 2.5;
        amplitude *= 0.5;
    }
    return value;
}

vec2 rotate(vec2 p, float a){
    float s =sin(a), c = cos(a);
    return mat2(c,-s,s,c)*p;
}

float stolb(vec3 p){
    p.xy = rotate(p.xy,0.2);
    float radius = 0.003+abs(p.y)*0.02;
    float d = length(p.xz)-radius;
    return d;
}

float sdTorus( vec3 p, vec2 t )
{
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

float GetDist(vec3 p,out float mat){
    float speed = 0.9;

    vec3 rotatedSpace = p;
    rotatedSpace.xz = rotate(rotatedSpace.xz, iTime * speed);

    p.xz = rotate(p.xz,iTime*speed);

    vec3 nPos = normalize(p);
    vec2 uvSphere = vec2(atan(nPos.z, nPos.x), nPos.y);


    vec2 qNoise = vec2(
        noise2D(uvSphere * 4.0 + iTime * 0.5),
        noise2D(uvSphere * 4.0 - iTime * 0.3)
    );

    float n = fbm2D(uvSphere * 20.0 + qNoise * 1.5 + iTime * 0.4) * 0.04;

    float planet = length(p)-(0.2+n);
    float jet = stolb(p);

    float smin = min(planet,jet);

    vec3 fieldSpace = rotatedSpace;

    fieldSpace.yz = rotate(fieldSpace.yz, 0.01);
    fieldSpace.xy = rotate(fieldSpace.xy, 0.2);

    float fields = 1e5;
    float Sphere = 1e5;

    const int rings = 8;
    for(int i=0;i<rings;i++){

        vec3 q =fieldSpace;

        q.xz = rotate(q.xz, float(i) * (6.14159 / float(rings)));
        q.yz = rotate(q.yz,1.6);
        q.xz = rotate(q.xz,-0.02);

        vec3 qt = q;
        qt.z *= 0.5;
        qt.x *= 0.6;

        float t = iTime * 5.0 + float(i) * 1.0;

        vec3 spherePos;
        spherePos.x = (-0.17 + cos(t) * 0.2) / 0.6;
        spherePos.y = 0.0;
        spherePos.z = (sin(t) * 0.2) / 0.5;

        float sphere = length(q-spherePos)-0.004;
        float magnit = sdTorus(qt+vec3(0.17,0.0,0.0),vec2(0.2,0.0001));

        fields = min(fields,magnit);
        Sphere = min(Sphere,sphere);
    }

    mat = 1.0;

    if(fields<smin){
        smin = fields;
        mat = 2.0;
    }
    if(Sphere<smin){
        smin = Sphere;
        mat = 3.0;
    }

    smin = min(smin,fields);

    return smin ;
}

float RayMarch(vec3 ro,vec3 rd,out float glow,out float res,out float mat){
    float d = 0.0;
    glow = 0.0;
    res =0.0;
    for(int i=0;i<STEPS;i++){
        vec3 p = ro +rd*d;
        float ds = GetDist(p,mat);
        d += ds;
        if(mat == 1.0){
           glow += 1.0/(abs(ds)+0.022);
        }
        if(mat == 2.0){
           glow += 1.0/(abs(ds)+0.19);
        }
        if(mat == 3.0){
           glow += 1.0/(abs(ds)+0.019);
        }

        if(d>MAX_DIST || abs(ds)<SURF_DIST) break;
    }
    return d;
}

mat2 rot(float a){
    float s=sin(a),c=cos(a);
    return mat2(c,-s,s,c);
}

vec3 R(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 f = normalize(l-p),
        r = normalize(cross(vec3(0,1,0), f)),
        u = cross(f,r),
        c = p+f*z,
        i = c + uv.x*r + uv.y*u,
        d = normalize(i-p);
    return d;
}
float get3dCubeStars(vec3 rd) {
    vec3 absRd = abs(rd);
    float maxAxis = max(absRd.x, max(absRd.y, absRd.z));

    vec2 uv = vec2(0.0);
    vec3 sectorId = vec3(0.0);

    if (maxAxis == absRd.x) {
        uv = rd.yz / rd.x;
        sectorId = vec3(sign(rd.x), 0.0, 0.0);
    } else if (maxAxis == absRd.y) {
        uv = rd.xz / rd.y;
        sectorId = vec3(0.0, sign(rd.y), 0.0);
    } else {
        uv = rd.xy / rd.z;
        sectorId = vec3(0.0, 0.0, sign(rd.z));
    }

    vec2 p = uv * 75.0;
    vec2 cellId = floor(p);
    vec2 gv = fract(p) - 0.5;

    vec2 finalId = cellId + sectorId.xy + sectorId.zz * 15.1;
    float h = hash2D(finalId);

    float star = 0.0;

    if(h > 0.88) {
        vec2 offset = vec2(hash2D(finalId + 0.35), hash2D(finalId + 0.72)) - 0.5;
        offset *= 0.6;

        float d = length(gv - offset);
        float size = 0.2 + hash2D(finalId * 1.5) * 0.08;
        star = exp(-35.0 * d / size);

        float twinkleSpeed = 0.3 + h * 0.8;

        float twinkle = sin(iTime * twinkleSpeed + h * 6.28) * 0.5 + 0.5;
        star *= mix(0.6, 1.0, twinkle);
    }
    return star;
}

vec3 fastCubeStars(vec3 rd) {
    float totalStars = get3dCubeStars(rd);
    return vec3(0.7, 0.85, 1.0) * totalStars * 100.5 * vec3(0.1, 0.5, 1.0);
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{

    vec2 uv = (fragCoord-.5*iResolution.xy)/iResolution.y;
    vec2 m = iMouse.xy/iResolution.xy;
    vec3 col = vec3(0);
    vec3 target = vec3(0,0.01,0);

    vec3 ro = vec3(0,-0.1,2.);
    ro.yz *= rot(-m.y*3.14+1.);
    ro.xz *= rot(-m.x*6.2831);

    vec3 rd = normalize(R(uv, ro, target, 1.2));

    vec3 Stars = fastCubeStars(rd);
    float glow = 0.0;
    float res =0.0;

    float mat ;
    float d = RayMarch(ro,rd,glow,res,mat);

    col = vec3(0.2,.4,.7) * glow *0.0045+0.09;

    if(d<50.) {
        if(mat ==1.0){
            col = vec3(0.2,.5,.7)*glow*0.02;
        }
        if(mat == 2.0){
            col = vec3(0.2,.5,.7)*glow*0.02;
        }
        if(mat == 3.0){
            col = vec3(0.2,.5,.7)*glow*0.02;
        }
    }else{
        col += Stars;
    }

    fragColor = vec4(col,1.0);
}
