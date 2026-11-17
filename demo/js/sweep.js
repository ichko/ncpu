// sweep.js — sliding-window NCA on a large canvas (WebGL2).
//
// A full canvas ("board") holds the persistent 16-channel state in 4 RGBA32F
// textures. Each simulation step advances a wh×ww window by (strideX, strideY)
// in serpentine order and applies ONE NCA update to just that window:
//
//   1. copyOut  — crop the window region out of the board into a small FBO;
//   2. update   — run the Sobel-perception + MLP + fire-mask step on the crop
//                 (zero-padded at the *window* edge, matching the PyTorch rule);
//   3. copyIn   — write the stepped window back into the board.
//
// The overlap (window − stride) is what lets settled state cross a window
// boundary, exactly like an STFT hop. The display pass renders channel 0 with
// the same viridis map as the training visualisations.

function b64ToF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

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

function updateFS(K, HID, channels, readOnly) {
  const PAD = (K - 1) / 2;
  const KK = K * K;
  const CV = channels / 4;             // vec4 groups
  const NFEAT = 3 * channels;          // 48
  const NPV = NFEAT / 4;               // 12
  const COMP = ['x', 'y', 'z', 'w'];
  const zeroLines = readOnly.map((c) => `  d${c >> 2}.${COMP[c & 3]}=0.0;`).join('\n');
  return `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D s0,s1,s2,s3, W1, W2, B1;
uniform float sob[${3 * KK}];
uniform vec2 uRes; uniform float uSeed; uniform float uFire;
uniform int uUpdateHead; uniform float uMaskSigma;
layout(location=0) out vec4 o0;
layout(location=1) out vec4 o1;
layout(location=2) out vec4 o2;
layout(location=3) out vec4 o3;
const int K=${K}, PAD=${PAD}, HID=${HID};
void main(){
  ivec2 P = ivec2(gl_FragCoord.xy);
  int Wd=int(uRes.x), Hd=int(uRes.y);
  vec4 pI[${CV}], pX[${CV}], pY[${CV}];
  for(int t=0;t<${CV};t++){ pI[t]=vec4(0.0); pX[t]=vec4(0.0); pY[t]=vec4(0.0); }
  for(int dy=0;dy<K;dy++) for(int dx=0;dx<K;dx++){
    int gx=P.x+dx-PAD, gy=P.y+dy-PAD;
    if(gx<0||gy<0||gx>=Wd||gy>=Hd) continue;
    ivec2 q=ivec2(gx,gy);
    vec4 a0=texelFetch(s0,q,0), a1=texelFetch(s1,q,0), a2=texelFetch(s2,q,0), a3=texelFetch(s3,q,0);
    int idx=dy*K+dx;
    float ki=sob[idx], kx=sob[K*K+idx], ky=sob[2*K*K+idx];
    pI[0]+=ki*a0; pI[1]+=ki*a1; pI[2]+=ki*a2; pI[3]+=ki*a3;
    pX[0]+=kx*a0; pX[1]+=kx*a1; pX[2]+=kx*a2; pX[3]+=kx*a3;
    pY[0]+=ky*a0; pY[1]+=ky*a1; pY[2]+=ky*a2; pY[3]+=ky*a3;
  }
  float perc[${NFEAT}];
  for(int c=0;c<${channels};c++){ int t=c/4, k=c-4*t;
    perc[c*3+0]=pI[t][k]; perc[c*3+1]=pX[t][k]; perc[c*3+2]=pY[t][k]; }
  vec4 Pv[${NPV}];
  for(int i=0;i<${NPV};i++) Pv[i]=vec4(perc[4*i],perc[4*i+1],perc[4*i+2],perc[4*i+3]);
  vec4 d0=vec4(0.0),d1=vec4(0.0),d2=vec4(0.0),d3=vec4(0.0);
  for(int j=0;j<HID;j++){
    float h=texelFetch(B1,ivec2(j,0),0).r;
    for(int i=0;i<${NPV};i++) h+=dot(texelFetch(W1,ivec2(i,j),0),Pv[i]);
    h=max(h,0.0);
    d0+=texelFetch(W2,ivec2(0,j),0)*h;
    d1+=texelFetch(W2,ivec2(1,j),0)*h;
    d2+=texelFetch(W2,ivec2(2,j),0)*h;
    d3+=texelFetch(W2,ivec2(3,j),0)*h;
  }
${zeroLines}
  float rnd=fract(sin(dot(gl_FragCoord.xy+vec2(uSeed,uSeed*1.7),vec2(12.9898,78.233)))*43758.5453);
  float fire = rnd < uFire ? 1.0 : 0.0;
  if(uUpdateHead == 1){
    vec2 offset = gl_FragCoord.xy - uRes * 0.5;
    // Even-sized windows have four central pixels; all four peak at exactly 1.
    fire *= exp(-(dot(offset,offset)-0.5)/(2.0*uMaskSigma*uMaskSigma));
  }
  vec4 c0=texelFetch(s0,P,0),c1=texelFetch(s1,P,0),c2=texelFetch(s2,P,0),c3=texelFetch(s3,P,0);
  o0=clamp(c0+fire*d0,-10.0,10.0);
  o1=clamp(c1+fire*d1,-10.0,10.0);
  o2=clamp(c2+fire*d2,-10.0,10.0);
  o3=clamp(c3+fire*d3,-10.0,10.0);
}`;
}

const COPY_FS = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D s0,s1,s2,s3;
uniform vec2 uOffset;
layout(location=0) out vec4 o0;
layout(location=1) out vec4 o1;
layout(location=2) out vec4 o2;
layout(location=3) out vec4 o3;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy) + ivec2(uOffset);
  ivec2 size = textureSize(s0, 0);
  if(any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size))){
    o0=vec4(0.0); o1=vec4(0.0); o2=vec4(0.0); o3=vec4(0.0); return;
  }
  o0 = texelFetch(s0,p,0); o1 = texelFetch(s1,p,0); o2 = texelFetch(s2,p,0); o3 = texelFetch(s3,p,0);
}`;

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
  vec2 uv = vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y);
  float v = texture(s0, uv).r;
  float t = clamp(v*0.5+0.5, 0.0, 1.0);
  frag = vec4(viridis(t), 1.0);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const kind = type === gl.VERTEX_SHADER ? 'VERTEX' : 'FRAGMENT';
    throw new Error(`[${kind}] log=${gl.getShaderInfoLog(s)}`);
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

// Window origins cover the board and include its far edges.
function axisPositions(extent, win, stride) {
  if (!Number.isInteger(stride) || stride < 1 || stride > win) throw new Error('Invalid stride');
  if (win > extent) throw new Error(`window ${win} > extent ${extent}`);
  const last = extent - win;
  if (last === 0) return [0];
  const pos = [];
  for (let p = 0; p <= last; p += stride) pos.push(p);
  if (pos[pos.length - 1] !== last) pos.push(last);
  return pos;
}

function shufflePositions(positions) {
  for (let i = positions.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [positions[i], positions[j]] = [positions[j], positions[i]];
  }
  return positions;
}

export function buildPositions(H, W, wh, ww, hstride, wstride, scanningPattern = 'raster') {
  if (!['raster', 'serpentine', 'column', 'random'].includes(scanningPattern)) throw new Error('Unknown scanning pattern');
  const rows = axisPositions(H, wh, hstride);
  const cols = axisPositions(W, ww, wstride);
  const positions = [];
  if (scanningPattern === 'column') {
    cols.forEach(x => rows.forEach(y => positions.push([y, x])));
    return positions;
  }
  rows.forEach((y, b) => {
    const cs = (scanningPattern === 'serpentine' && b % 2 !== 0) ? cols.slice().reverse() : cols;
    cs.forEach((x) => positions.push([y, x]));
  });
  return scanningPattern === 'random' ? shufflePositions(positions) : positions;
}

export class SweepSim {
  constructor(canvas, model, opts) {
    if (model.channels !== 16 || model.padding !== 'zeros') throw new Error('Engine requires 16 channels and zero padding');
    this.model = model;
    this.W = opts.W; this.H = opts.H;
    this.winW = opts.winW; this.winH = opts.winH;
    this.contextPadding = opts.contextPadding ?? 3;
    this.updateHead = opts.updateHead || 'uniform';
    this.maskSigma = opts.maskSigma ?? 8;
    if (!Number.isInteger(this.contextPadding) || this.contextPadding < 0 || this.contextPadding > 16) throw new Error('Context padding must be 0–16');
    this.strideX = opts.strideX; this.strideY = opts.strideY;
    this.scanningPattern = opts.scanningPattern || 'raster';
    this.positions = buildPositions(this.H, this.W, this.winH, this.winW, this.strideY, this.strideX, this.scanningPattern);
    this.cursor = 0;
    this.windowUpdates = 0;

    const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: false });
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl;
    canvas.width = this.W; canvas.height = this.H;
    gl.getExtension('EXT_color_buffer_float');
    gl.getExtension('EXT_color_buffer_half_float');
    gl.getExtension('OES_texture_float_linear');

    this.updateProg = program(gl, QUAD_VS, updateFS(model.kernel, model.hidden, model.channels, model.read_only));
    this.copyProg = program(gl, QUAD_VS, COPY_FS);
    this.displayProg = program(gl, QUAD_VS, DISPLAY_FS);

    this.quad = gl.createVertexArray();
    gl.bindVertexArray(this.quad);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    this.sob = b64ToF32(model.sobel.data);
    this.texW1 = this._dataTex(model.W1.w, model.W1.h, gl.RGBA32F, gl.RGBA, gl.FLOAT, b64ToF32(model.W1.data));
    this.texW2 = this._dataTex(model.W2.w, model.W2.h, gl.RGBA32F, gl.RGBA, gl.FLOAT, b64ToF32(model.W2.data));
    this.texB1 = this._dataTex(model.b1.n, 1, gl.R32F, gl.RED, gl.FLOAT, b64ToF32(model.b1.data));

    this.stInternal = gl.RGBA32F; this.stType = gl.FLOAT;
    this.board = this._stateFBO(this.W, this.H);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      this._disposeState(this.board);
      this.stInternal = gl.RGBA16F; this.stType = gl.HALF_FLOAT;
      this.board = this._stateFBO(this.W, this.H);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error('float render targets unavailable');
    }
    this.winA = this._stateFBO(this.winW + 2*this.contextPadding, this.winH + 2*this.contextPadding);
    this.winB = this._stateFBO(this.winW + 2*this.contextPadding, this.winH + 2*this.contextPadding);

    this.seedVal = Math.random() * 1000;
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

  _stateFBO(w, h) {
    const gl = this.gl;
    const texs = [];
    for (let i = 0; i < 4; i++) texs.push(this._dataTex(w, h, this.stInternal, gl.RGBA, this.stType, null));
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    for (let i = 0; i < 4; i++)
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, texs[i], 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2, gl.COLOR_ATTACHMENT3]);
    return { texs, fbo };
  }
  _disposeState(S) { const gl = this.gl; S.texs.forEach((t) => gl.deleteTexture(t)); gl.deleteFramebuffer(S.fbo); }

  // Upload a fresh board: input (ch0/ch1) + selector (ch2) float arrays.
  uploadBoard(inputImg, selectorImg) {
    const gl = this.gl, N = this.W * this.H;
    const t0 = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) {
      t0[i * 4] = inputImg[i];
      t0[i * 4 + 1] = inputImg[i];
      t0[i * 4 + 2] = selectorImg[i];
      t0[i * 4 + 3] = this.model.position_channel === 3
        ? ((i % this.W) % this.model.window + Math.floor(i / this.W) % this.model.window) / (this.model.window - 1) - 1
        : 0.0;
    }
    const zero = new Float32Array(N * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.board.fbo);
    this._upload(this.board.texs[0], t0);
    this._upload(this.board.texs[1], zero);
    this._upload(this.board.texs[2], zero);
    this._upload(this.board.texs[3], zero);
    this.cursor = 0;
    this.windowUpdates = 0;
    this.seedVal = Math.random() * 1000;
  }

  _upload(tex, data) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    if (this.stType === gl.HALF_FLOAT) data = f32ToF16(data);
    gl.texImage2D(gl.TEXTURE_2D, 0, this.stInternal, this.W, this.H, 0, gl.RGBA, this.stType, data);
  }

  _bindInputs(prog, texs) {
    const gl = this.gl;
    for (let i = 0; i < 4; i++) {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, texs[i]);
      gl.uniform1i(gl.getUniformLocation(prog, 's' + i), i);
    }
  }

  stepWindow(innerSteps = 1, advance = true) {
    // Refresh halo from persistent board after each update, matching tick().
    if (this.contextPadding && innerSteps > 1) {
      for (let i = 0; i < innerSteps; i++) this.stepWindow(1, advance && i === innerSteps-1);
      return;
    }
    const gl = this.gl;
    if (this.positions.length === 0) return;
    const [y0, x0] = this.positions[this.cursor % this.positions.length];
    const P = this.updateProg;
    const pad = this.contextPadding, updateW = this.winW + 2*pad, updateH = this.winH + 2*pad;

    // 1) copyOut: board -> winA (offset = window origin)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.winA.fbo);
    gl.viewport(0, 0, updateW, updateH);
    gl.useProgram(this.copyProg);
    gl.bindVertexArray(this.quad);
    this._bindInputs(this.copyProg, this.board.texs);
    gl.uniform2f(gl.getUniformLocation(this.copyProg, 'uOffset'), x0-pad, y0-pad);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // 2) innerSteps NCA updates, ping-ponging winA <-> winB
    let src = this.winA, dst = this.winB;
    for (let s = 0; s < innerSteps; s++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
      gl.viewport(0, 0, updateW, updateH);
      gl.useProgram(P);
      gl.bindVertexArray(this.quad);
      this._bindInputs(P, src.texs);
      gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, this.texW1); gl.uniform1i(gl.getUniformLocation(P, 'W1'), 4);
      gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, this.texW2); gl.uniform1i(gl.getUniformLocation(P, 'W2'), 5);
      gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, this.texB1); gl.uniform1i(gl.getUniformLocation(P, 'B1'), 6);
      gl.uniform1fv(gl.getUniformLocation(P, 'sob'), this.sob);
      gl.uniform2f(gl.getUniformLocation(P, 'uRes'), updateW, updateH);
      gl.uniform1f(gl.getUniformLocation(P, 'uSeed'), this.seedVal);
      gl.uniform1f(gl.getUniformLocation(P, 'uFire'), this.model.fire_rate);
      gl.uniform1i(gl.getUniformLocation(P, 'uUpdateHead'), this.updateHead === 'gaussian' ? 1 : 0);
      gl.uniform1f(gl.getUniformLocation(P, 'uMaskSigma'), this.maskSigma);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      [src, dst] = [dst, src];
      this.seedVal = (this.seedVal + 1.618) % 4096;
    }

    // 3) copyIn: src (final window state) -> board (scissored to window region)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.board.fbo);
    gl.viewport(x0, y0, this.winW, this.winH);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(x0, y0, this.winW, this.winH);
    gl.useProgram(this.copyProg);
    gl.bindVertexArray(this.quad);
    this._bindInputs(this.copyProg, src.texs);
    gl.uniform2f(gl.getUniformLocation(this.copyProg, 'uOffset'), pad-x0, pad-y0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.disable(gl.SCISSOR_TEST);

    if (advance) {
      this.cursor = (this.cursor + 1) % this.positions.length;
      if (this.cursor === 0 && this.scanningPattern === 'random') shufflePositions(this.positions);
      this.windowUpdates = 0;
    }
  }

  // Bound work per animation frame; carry state and dwell count between frames.
  tick(updates = 2, dwell = 8) {
    for (let i = 0; i < updates; i++) {
      this.windowUpdates++;
      const advance = this.windowUpdates >= dwell;
      this.stepWindow(1, advance);
      if (advance) this.windowUpdates = 0;
    }
  }

  setStride(x, y) {
    this.strideX = x; this.strideY = y;
    this.positions = buildPositions(this.H, this.W, this.winH, this.winW, y, x, this.scanningPattern);
    this.cursor = 0; this.windowUpdates = 0;
  }

  setScanningPattern(pattern) {
    this.scanningPattern = pattern;
    this.setStride(this.strideX, this.strideY);
  }

  setContextPadding(padding) {
    if (!Number.isInteger(padding) || padding < 0 || padding > 16) throw new Error('Context padding must be 0–16');
    this._disposeState(this.winA); this._disposeState(this.winB);
    this.contextPadding = padding;
    this.winA = this._stateFBO(this.winW + 2*padding, this.winH + 2*padding);
    this.winB = this._stateFBO(this.winW + 2*padding, this.winH + 2*padding);
  }

  currentWindow() {
    return this.positions[this.cursor % this.positions.length];
  }

  draw() {
    const gl = this.gl, P = this.displayProg;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.W, this.H);
    gl.useProgram(P);
    gl.bindVertexArray(this.quad);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.board.texs[0]);
    gl.uniform1i(gl.getUniformLocation(P, 's0'), 0);
    gl.uniform2f(gl.getUniformLocation(P, 'uRes'), this.W, this.H);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  dispose() {
    const gl = this.gl;
    [this.texW1, this.texW2, this.texB1].forEach((t) => gl.deleteTexture(t));
    for (const S of [this.board, this.winA, this.winB]) this._disposeState(S);
    gl.deleteProgram(this.updateProg);
    gl.deleteProgram(this.copyProg);
    gl.deleteProgram(this.displayProg);
    gl.deleteVertexArray(this.quad);
  }
}
