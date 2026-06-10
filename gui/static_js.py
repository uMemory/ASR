"""JavaScript snippets injected into the Gradio page."""

PLAY_JS = """<script>
let activeTimeout=null;
let activeAudio=null;
function _audioHasSource(a){
  if(!a)return false;
  if(a.currentSrc || a.src || a.getAttribute('src'))return true;
  var s=a.querySelector ? a.querySelector('source[src]') : null;
  return !!(s && s.getAttribute('src'));
}
function _firstAudioIn(root){
  if(!root || !root.querySelector)return null;
  var list=root.querySelectorAll('audio,video');
  for(var i=0;i<list.length;i++){if(_audioHasSource(list[i]))return list[i];}
  return null;
}
function _getAudio(audioId, trigger){
  if(audioId){
    if(audioId==='history-audio-player' || audioId==='main-audio-player'){
      var preferred=document.getElementById(audioId);
      var preferredAudio=_firstAudioIn(preferred);
      if(preferredAudio)return preferredAudio;
    }
    var direct=document.getElementById(audioId);
    if(_audioHasSource(direct))return direct;
    var nested=_firstAudioIn(direct);
    if(nested)return nested;
  }
  if(trigger && trigger.closest){
    var panel=trigger.closest('.file-panel,.history-panel,.transcript-file');
    var scoped=_firstAudioIn(panel);
    if(scoped)return scoped;
  }
  var playerIds=['main-audio-player','history-audio-player'];
  for(var p=0;p<playerIds.length;p++){
    var wrap=document.getElementById(playerIds[p]);
    var a=_firstAudioIn(wrap);
    if(a)return a;
  }
  var all=document.querySelectorAll('audio,video');
  for(var i=0;i<all.length;i++){if(_audioHasSource(all[i]))return all[i];}
  return all.length>0?all[0]:null;
}
function _seekAndPlay(a,s,e){
  if(activeTimeout)clearTimeout(activeTimeout);
  document.querySelectorAll('audio').forEach(x=>{
    if(x!==a){try{x.pause();}catch(err){}}
  });
  if(activeAudio && activeAudio!==a){try{activeAudio.pause();}catch(err){}}
  activeAudio=a;
  const duration=Math.max(0.1,(e-s))*1000+300;
  const start=function(){
    try{a.currentTime=s;}catch(err){}
    const p=a.play();
    if(p&&p.catch)p.catch(err=>console.log('audio play blocked or failed',err));
    activeTimeout=setTimeout(()=>{try{a.pause();}catch(err){}},duration);
  };
  if(a.readyState>=1){start();return;}
  a.preload='auto';
  a.addEventListener('loadedmetadata',start,{once:true});
  a.addEventListener('canplay',start,{once:true});
  try{a.load();}catch(err){}
}
function playSeg(s,e,audioId,trigger){let a=_getAudio(audioId,trigger);if(!a){console.log('no audio found for',audioId);return;}
_seekAndPlay(a,s,e);}
function playRtSeg(c,o,d){let a=document.getElementById('rt-audio-'+c);if(!a){a=_getAudio(null,null);if(!a)return;}
_seekAndPlay(a,o,o+d);}
window.playSeg=playSeg;
window.playRtSeg=playRtSeg;
document.addEventListener('click',function(ev){
  var line=ev.target && ev.target.closest ? ev.target.closest('.seg-line[data-play-start]') : null;
  if(!line)return;
  if(line.getAttribute('onclick'))return;
  ev.preventDefault();
  ev.stopPropagation();
  var s=parseFloat(line.dataset.playStart);
  var e=parseFloat(line.dataset.playEnd);
  if(!isFinite(s)||!isFinite(e))return;
  var audioId=line.dataset.audioId||null;
  playSeg(s,e,audioId,line);
},true);
function applyRename(){
const map={};
document.querySelectorAll('.speaker-rename').forEach(inp=>{
  if(!inp.value.trim())return;
  const fid=inp.dataset.fileIndex||'';
  map[fid+'|'+inp.dataset.speaker]=inp.value.trim();
});
document.querySelectorAll('.seg-line b').forEach(el=>{
var raw=el.getAttribute('data-speaker')||el.textContent.trim();
var fid=el.getAttribute('data-file-index')||'';
el.textContent=map[fid+'|'+raw]||raw;
});
}
window.applyRename=applyRename;
</script>"""


MIC_JS = """<script>
let micWs=null,micStream=null,micCtx=null,micProc=null,micRunning=false,micPaused=false,micBuffer=[],micChunkId=0,micSampleRate=16000,micChunkDuration=10;
function encodeWAV(samples,sr){const buf=new ArrayBuffer(44+samples.length*2);const v=new DataView(buf);function ws(o,s){for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i))}ws(0,'RIFF');v.setUint32(4,36+samples.length*2,true);ws(8,'WAVE');ws(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,sr,true);v.setUint32(28,sr*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);ws(36,'data');v.setUint32(40,samples.length*2,true);for(let i=0;i<samples.length;i++){const s=Math.max(-1,Math.min(1,samples[i]));v.setInt16(44+i*2,s<0?s*0x8000:s*0x7FFF,true)}return new Uint8Array(buf)}
function ab2b64(buf){let b='';const u=new Uint8Array(buf);for(let i=0;i<u.length;i++)b+=String.fromCharCode(u[i]);return btoa(b)}
function connectWs(){if(micWs&&(micWs.readyState===WebSocket.OPEN||micWs.readyState===WebSocket.CONNECTING))return;const proto=location.protocol==='https:'?'wss:':'ws:';micWs=new WebSocket(proto+'//'+location.hostname+':7861');micWs.onopen=()=>{micWs.send(JSON.stringify({type:'init',chunk_duration:micChunkDuration}))};micWs.onmessage=()=>{};micWs.onclose=()=>{}}
function sendMicChunk(samples){if(!samples||samples.length<micSampleRate*0.3)return;if(micWs&&micWs.readyState===WebSocket.OPEN){micChunkId++;micWs.send(JSON.stringify({type:'audio',data:ab2b64(encodeWAV(samples,micSampleRate)),chunk_id:micChunkId}))}}
function startAudioStream(stream){connectWs();micStream=stream;micCtx=new AudioContext({sampleRate:micSampleRate});const src=micCtx.createMediaStreamSource(micStream);micProc=micCtx.createScriptProcessor(4096,1,1);micBuffer=[];micChunkId=0;micRunning=true;micPaused=false;micStream.getTracks().forEach(t=>{t.onended=()=>{stopMic()}});micProc.onaudioprocess=function(e){if(!micRunning||micPaused)return;const inp=e.inputBuffer.getChannelData(0);for(let i=0;i<inp.length;i++)micBuffer.push(inp[i]);while(micBuffer.length>=micSampleRate*micChunkDuration){sendMicChunk(new Float32Array(micBuffer.splice(0,micSampleRate*micChunkDuration)))}};src.connect(micProc);micProc.connect(micCtx.destination);return 'recording'}
async function startMic(){if(micRunning){micPaused=false;return 'recording'}try{const stream=await navigator.mediaDevices.getUserMedia({audio:{sampleRate:micSampleRate,channelCount:1,echoCancellation:true}});return startAudioStream(stream)}catch(e){console.log('mic capture failed',e);return 'error'}}
async function startSystemAudio(){if(micRunning){micPaused=false;return 'recording'}try{const stream=await navigator.mediaDevices.getDisplayMedia({video:true,audio:{sampleRate:micSampleRate,channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false}});if(stream.getAudioTracks().length===0){stream.getTracks().forEach(t=>t.stop());return 'noaudio'}return startAudioStream(stream)}catch(e){console.log('system audio capture failed',e);return 'error'}}
function pauseMic(){if(!micRunning)return 'stopped';micPaused=!micPaused;return micPaused?'paused':'recording'}
function stopMic(){micRunning=false;micPaused=false;if(micBuffer&&micBuffer.length>micSampleRate*0.3){sendMicChunk(new Float32Array(micBuffer.splice(0,micBuffer.length)))}if(micProc){try{micProc.disconnect()}catch(e){} micProc=null}if(micStream){micStream.getTracks().forEach(t=>t.stop());micStream=null}if(micCtx){micCtx.close();micCtx=null}if(micWs){setTimeout(()=>{try{micWs.close()}catch(e){} micWs=null},800)}}
document.addEventListener('focusin', function(e){
  const target=e.target;
  if(!target || !target.closest || !target.closest('#output-dir-input')) return;
  if(target.tagName==='TEXTAREA' || target.tagName==='INPUT'){
    setTimeout(()=>target.select(), 0);
  }
});
let rtRawShown='',rtRawTarget='',rtRawTimer=null;
function _rtRawSource(){
  return document.querySelector('#rt-raw-source textarea') || document.querySelector('#rt-raw-source input');
}
function _rtRawBox(){
  return document.getElementById('rt-raw-stream-box');
}
function _rtRenderRaw(text){
  const box=_rtRawBox(); if(!box)return;
  box.textContent=text||'等待 ASR 直接结果...';
  box.scrollTop=box.scrollHeight;
}
function _rtStepRaw(){
  if(rtRawShown===rtRawTarget){rtRawTimer=null;return;}
  const delta=rtRawTarget.length-rtRawShown.length;
  if(delta>0){
    let next=rtRawShown.length+1;
    while(next<rtRawTarget.length && !/\\s/.test(rtRawTarget.charAt(next-1)))next++;
    while(next<rtRawTarget.length && /\\s/.test(rtRawTarget.charAt(next)))next++;
    next=Math.min(next, rtRawShown.length+28, rtRawTarget.length);
    rtRawShown=rtRawTarget.slice(0,next);
  }else{
    rtRawShown=rtRawTarget;
  }
  _rtRenderRaw(rtRawShown);
  rtRawTimer=setTimeout(_rtStepRaw,42);
}
function _rtUpdateRawStream(){
  const src=_rtRawSource(); if(!src)return;
  const next=src.value||'';
  if(next===rtRawTarget)return;
  rtRawTarget=next;
  if(!rtRawTimer)_rtStepRaw();
}
setInterval(_rtUpdateRawStream,160);
</script>"""
