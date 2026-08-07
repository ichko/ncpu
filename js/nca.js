// nca.js — runs a trained Neural Cellular Automaton in the browser with WebGL2.
//
// The state is 16 channels packed into 4 RGBA32F textures. Each simulation step
// is one fragment-shader pass with 4 render targets (MRT): it reads the k×k
// neighbourhood, applies the fixed Sobel/identity perception, then the learned
// 48->128->16 per-pixel MLP, zeroes the read-only channel, applies the
// stochastic fire mask, and writes the clamped new state. All math verified
// against the PyTorch model to ~1e-6.

function b64ToF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

// Float32Array -> Uint16Array of IEEE half-floats (for the 16F fallback path)
const _f32 = new Float32Array(1), _i32 = new Int32Array(_f32.buffer);
function f32ToF16(arr) {
  const out = new Uint16Array(arr.length);
  for (let k = 0; k < arr.length; k++) {
    _f32[0] = arr[k]; const x = _i32[0];
    let bits = (x >> 16) & 0x8000; let m = (x >> 12) & 0x07ff; const e = (x >> 23) & 0xff;
    if (e < 103) { out[k] = bits; continue; }
    if (e > 142) { bits |= 0x7c00; bits |= (e === 255 ? 0 : 1) && (x & 0x007fffff); out[k] = bits; continue; }
    if (e < 113) { m |= 0x0800; bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1); out[k] = bits; continue; }
    bits |= ((e - 112) << 10) | (m >> 1); bits += m & 1; out[k] = bits;
  }
  return out;
}

const QUAD_VS = `#version 300 es
in vec2 p; void main(){ gl_Position = vec4(p,0.0,1.0); }`;

function updateFS(K) {
  const PAD = (K - 1) / 2;
  const KK = K * K;
  return `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D s0,s1,s2,s3, W1, W2, B1;
uniform float sob[${3 * KK}];
uniform vec2 uRes; uniform float uSeed; uniform float uFire;
layout(location=0) out vec4 o0;
layout(location=1) out vec4 o1;
layout(location=2) out vec4 o2;
layout(location=3) out vec4 o3;
const int K=${K}, PAD=${PAD}, HID=128;
void main(){
  ivec2 P = ivec2(gl_FragCoord.xy);
  int Wd=int(uRes.x), Hd=int(uRes.y);
  vec4 pI[4], pX[4], pY[4];
  for(int t=0;t<4;t++){ pI[t]=vec4(0.0); pX[t]=vec4(0.0); pY[t]=vec4(0.0); }
  for(int dy=0;dy<K;dy++) for(int dx=0;dx<K;dx++){
    int gx=P.x+dx-PAD, gy=P.y+dy-PAD;
    if(gx<0||gy<0||gx>=Wd||gy>=Hd) continue;      // zero padding
    ivec2 q=ivec2(gx,gy);
    vec4 a0=texelFetch(s0,q,0), a1=texelFetch(s1,q,0), a2=texelFetch(s2,q,0), a3=texelFetch(s3,q,0);
    int idx=dy*K+dx;
    float ki=sob[idx], kx=sob[K*K+idx], ky=sob[2*K*K+idx];
    pI[0]+=ki*a0; pI[1]+=ki*a1; pI[2]+=ki*a2; pI[3]+=ki*a3;
    pX[0]+=kx*a0; pX[1]+=kx*a1; pX[2]+=kx*a2; pX[3]+=kx*a3;
    pY[0]+=ky*a0; pY[1]+=ky*a1; pY[2]+=ky*a2; pY[3]+=ky*a3;
  }
  float perc[48];
  for(int c=0;c<16;c++){ int t=c/4, k=c-4*t;
    perc[c*3+0]=pI[t][k]; perc[c*3+1]=pX[t][k]; perc[c*3+2]=pY[t][k]; }
  vec4 Pv[12];
  for(int i=0;i<12;i++) Pv[i]=vec4(perc[4*i],perc[4*i+1],perc[4*i+2],perc[4*i+3]);
  vec4 d0=vec4(0.0),d1=vec4(0.0),d2=vec4(0.0),d3=vec4(0.0);
  for(int j=0;j<HID;j++){
    float h=texelFetch(B1,ivec2(j,0),0).r;
    for(int i=0;i<12;i++) h+=dot(texelFetch(W1,ivec2(i,j),0),Pv[i]);
    h=max(h,0.0);
    d0+=texelFetch(W2,ivec2(0,j),0)*h;
    d1+=texelFetch(W2,ivec2(1,j),0)*h;
    d2+=texelFetch(W2,ivec2(2,j),0)*h;
    d3+=texelFetch(W2,ivec2(3,j),0)*h;
  }
  d0.y=0.0;                                        // channel 1 is read-only
  float rnd=fract(sin(dot(gl_FragCoord.xy+vec2(uSeed,uSeed*1.7),vec2(12.9898,78.233)))*43758.5453);
  float fire = rnd < uFire ? 1.0 : 0.0;
  vec4 c0=texelFetch(s0,P,0),c1=texelFetch(s1,P,0),c2=texelFetch(s2,P,0),c3=texelFetch(s3,P,0);
  o0=clamp(c0+fire*d0,-10.0,10.0);
  o1=clamp(c1+fire*d1,-10.0,10.0);
  o2=clamp(c2+fire*d2,-10.0,10.0);
  o3=clamp(c3+fire*d3,-10.0,10.0);
}`;
}

// viridis colormap (polynomial fit), matching the training visualisations
// (mediapy show_images with cmap="viridis", vmin=-1, vmax=1).
const DISPLAY_FS = `#version 300 es
precision highp float; precision highp sampler2D;
uniform sampler2D s0; uniform vec2 uRes;
out vec4 frag;
vec3 viridis(float t){
  const vec3 c0=vec3(0.2777273272234177,0.005407344544966578,0.3340998053353061);
  const vec3 c1=vec3(0.1050930431085774,1.404613529898575,1.384590162594685);
  const vec3 c2=vec3(-0.3308618287255563,0.214847559468213,0.09509516302823659);
  const vec3 c3=vec3(-4.634230498983486,-5.799100973351585,-19.33244095627987);
  const vec3 c4=vec3(6.228269936347081,14.17993336680509,56.69055260068105);
  const vec3 c5=vec3(4.776384997670288,-13.74514537774601,-65.35303263337234);
  const vec3 c6=vec3(-5.435455855934631,4.645852612178535,26.3124352495832);
  return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
void main(){
  vec2 uv = vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y); // row 0 at top
  float v = texture(s0, uv).r;              // channel 0
  float t = clamp(v*0.5+0.5, 0.0, 1.0);     // -1..1 -> 0..1
  frag = vec4(viridis(t), 1.0);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const kind = type === gl.VERTEX_SHADER ? 'VERTEX' : 'FRAGMENT';
    throw new Error(`[${kind}] err=${gl.getError()} log=${gl.getShaderInfoLog(s)}`);
  }
  return s;
}
function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

export class NCASim {
  constructor(canvas, model) {
    this.model = model;
    this.W = model.W; this.H = model.H;
    const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: false });
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl;
    canvas.width = this.W; canvas.height = this.H;
    // enable float rendering extensions (return value is unreliable across
    // engines; we decide the usable format by framebuffer completeness below)
    gl.getExtension('EXT_color_buffer_float');
    gl.getExtension('EXT_color_buffer_half_float');
    gl.getExtension('OES_texture_float_linear');

    this.updateProg = program(gl, QUAD_VS, updateFS(model.kernel));
    this.displayProg = program(gl, QUAD_VS, DISPLAY_FS);

    // fullscreen quad
    this.quad = gl.createVertexArray();
    gl.bindVertexArray(this.quad);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    // sobel uniform data
    this.sob = b64ToF32(model.sobel.data);

    // weights are only ever sampled (not rendered to), so always full 32-bit
    this.texW1 = this._dataTex(model.W1.w, model.W1.h, gl.RGBA32F, gl.RGBA, gl.FLOAT, b64ToF32(model.W1.data));
    this.texW2 = this._dataTex(model.W2.w, model.W2.h, gl.RGBA32F, gl.RGBA, gl.FLOAT, b64ToF32(model.W2.data));
    this.texB1 = this._dataTex(model.b1.n, 1, gl.R32F, gl.RED, gl.FLOAT, b64ToF32(model.b1.data));

    // pick the highest-precision renderable state format this engine supports
    this.stInternal = gl.RGBA32F; this.stType = gl.FLOAT;
    this.A = this._stateFBO();
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      this._disposeState(this.A);
      this.stInternal = gl.RGBA16F; this.stType = gl.HALF_FLOAT;
      this.A = this._stateFBO();
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error('float render targets unavailable');
    }
    this.B = this._stateFBO();
    this.cur = this.A;

    this.seedVal = Math.random() * 1000;
    this.readBuf = new Float32Array(this.W * this.H * 4);
  }

  _dataTex(w, h, internal, format, type, data) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data);
    return t;
  }

  _stateFBO() {
    const gl = this.gl;
    const texs = [];
    for (let i = 0; i < 4; i++) texs.push(this._dataTex(this.W, this.H, this.stInternal, gl.RGBA, this.stType, null));
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    for (let i = 0; i < 4; i++)
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, texs[i], 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2, gl.COLOR_ATTACHMENT3]);
    return { texs, fbo };
  }
  _disposeState(S) { const gl = this.gl; S.texs.forEach((t) => gl.deleteTexture(t)); gl.deleteFramebuffer(S.fbo); }

  // seed: img = Float32Array(H*W) channel-0 image; sets ch0 and ch1, rest 0
  seed(img) {
    const gl = this.gl, N = this.W * this.H;
    const t0 = new Float32Array(N * 4); // ch0..3
    for (let i = 0; i < N; i++) { t0[i*4] = img[i]; t0[i*4+1] = img[i]; } // ch0=ch1=input
    const zero = new Float32Array(N * 4);
    for (const S of [this.A, this.B]) {
      this._upload(S.texs[0], t0);
      this._upload(S.texs[1], zero);
      this._upload(S.texs[2], zero);
      this._upload(S.texs[3], zero);
    }
    this.cur = this.A;
    this.seedVal = Math.random() * 1000;
  }

  _upload(tex, data) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    if (this.stType === gl.HALF_FLOAT) data = f32ToF16(data);
    gl.texImage2D(gl.TEXTURE_2D, 0, this.stInternal, this.W, this.H, 0, gl.RGBA, this.stType, data);
  }

  step(fireRate) {
    const gl = this.gl, src = this.cur, dst = (this.cur === this.A) ? this.B : this.A;
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
    gl.viewport(0, 0, this.W, this.H);
    gl.useProgram(this.updateProg);
    gl.bindVertexArray(this.quad);
    const P = this.updateProg;
    for (let i = 0; i < 4; i++) {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, src.texs[i]);
      gl.uniform1i(gl.getUniformLocation(P, 's' + i), i);
    }
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, this.texW1); gl.uniform1i(gl.getUniformLocation(P, 'W1'), 4);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, this.texW2); gl.uniform1i(gl.getUniformLocation(P, 'W2'), 5);
    gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, this.texB1); gl.uniform1i(gl.getUniformLocation(P, 'B1'), 6);
    gl.uniform1fv(gl.getUniformLocation(P, 'sob'), this.sob);
    gl.uniform2f(gl.getUniformLocation(P, 'uRes'), this.W, this.H);
    gl.uniform1f(gl.getUniformLocation(P, 'uSeed'), this.seedVal);
    gl.uniform1f(gl.getUniformLocation(P, 'uFire'), fireRate ?? this.model.fire_rate);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.cur = dst;
    this.seedVal = (this.seedVal + 1.618) % 4096;
  }

  draw() {
    const gl = this.gl, P = this.displayProg;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.W, this.H);
    gl.useProgram(P);
    gl.bindVertexArray(this.quad);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.cur.texs[0]);
    gl.uniform1i(gl.getUniformLocation(P, 's0'), 0);
    gl.uniform2f(gl.getUniformLocation(P, 'uRes'), this.W, this.H);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  dispose() {
    const gl = this.gl;
    [this.texW1, this.texW2, this.texB1].forEach((t) => gl.deleteTexture(t));
    for (const S of [this.A, this.B]) {
      S.texs.forEach((t) => gl.deleteTexture(t));
      gl.deleteFramebuffer(S.fbo);
    }
    gl.deleteProgram(this.updateProg);
    gl.deleteProgram(this.displayProg);
    gl.deleteVertexArray(this.quad);
  }

  // read channel 0 -> Float32Array(H*W), row 0 = top (flips GL's bottom-origin)
  readChannel0() {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.cur.fbo);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, this.W, this.H, gl.RGBA, gl.FLOAT, this.readBuf);
    // texImage2D upload, texelFetch, and readPixels all share the same
    // data-space row indexing, so no vertical flip is needed here.
    const out = new Float32Array(this.W * this.H);
    for (let i = 0; i < out.length; i++) out[i] = this.readBuf[i * 4];
    return out;
  }
}
