// Rigid placement of recorded particle steps. Coordinates are in millimeters.
export const IDENTITY = Object.freeze([1,0,0, 0,1,0, 0,0,1]);
export function rotateVector(matrix, vector){
  return [0,3,6].map(row=>matrix[row]*vector[0]+matrix[row+1]*vector[1]+matrix[row+2]*vector[2]);
}
export function rotatePoint(matrix, point, pivot){
  return rotateVector(matrix,point.map((x,i)=>x-pivot[i])).map((x,i)=>x+pivot[i]);
}
export function poseEvent(rows, {azimuth=0,elevation=90}={}){
  if(!rows.length)throw Error('Cannot aim an empty event');
  if(!Number.isFinite(azimuth)||!Number.isFinite(elevation)||elevation < -90||elevation > 90)throw Error('Invalid particle direction');
  azimuth=((azimuth+180)%360+360)%360-180;
  // Preserve the exported +z pose exactly, while retaining the selected azimuth.
  const pivot=rows.reduce((a,b)=>b[3]<a[3]?b:a).slice(0,3);
  const aim={azimuth,elevation};
  if(elevation===90)return {rows,rotation:IDENTITY,pivot,axis:[0,0,1],aim};
  const snap=x=>Math.abs(x)<1e-15?0:Math.abs(1-Math.abs(x))<1e-15?Math.sign(x):x;
  const phi=azimuth*Math.PI/180,el=elevation*Math.PI/180;
  const a=snap(Math.cos(phi)),b=snap(Math.sin(phi)),c=snap(Math.sin(el)),s=snap(Math.cos(el));
  // Rz(phi) Ry(pi/2 - elevation) Rz(-phi): a proper rotation taking +z to axis.
  const rotation=[c*a*a+b*b,(c-1)*a*b,s*a, (c-1)*a*b,c*b*b+a*a,s*b, -s*a,-s*b,c];
  const posed=rows.map(row=>{
    const result=row.slice();
    for(const offset of [0,4])result.splice(offset,3,...rotatePoint(rotation,row.slice(offset,offset+3),pivot));
    result.splice(8,3,...rotateVector(rotation,row.slice(8,11)));
    return result;
  });
  return {rows:posed,rotation,pivot,axis:[s*a,s*b,c],aim};
}
