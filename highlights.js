'use strict';
// Play only videos visible in the page and within the horizontally scrolling rails.
const videos = [...document.querySelectorAll('video')];
const visible = new Set();
const manuallyPaused = new WeakSet();
const automaticPause = new WeakSet();
function updateVideo(video) {
 if (visible.has(video) && !document.hidden && !manuallyPaused.has(video)) {
  video.muted = true;
  video.play().catch(() => {});
 } else if (!video.paused) {
  automaticPause.add(video); video.pause();
 }
}
videos.forEach(video => {
 video.autoplay = false;
 video.addEventListener('pause', () => {if (automaticPause.has(video)) automaticPause.delete(video); else manuallyPaused.add(video);});
 video.addEventListener('play', () => manuallyPaused.delete(video));
});
if ('IntersectionObserver' in window) {
 const observer = new IntersectionObserver(entries => entries.forEach(entry => {
  if (entry.isIntersecting && entry.intersectionRatio >= .25) visible.add(entry.target); else visible.delete(entry.target);
  updateVideo(entry.target);
 }), {threshold:[0,.25,.5]});
 videos.forEach(video => observer.observe(video));
} else {videos.forEach(video => {visible.add(video);updateVideo(video);});}
document.addEventListener('visibilitychange',()=>videos.forEach(updateVideo));

const sceneSelect = document.querySelector('#history-scene');
const visits = document.querySelector('#visit-buttons');
let selectedVisit = 0;
function renderHistory() {
 const scene = sceneSelect.value;
 const frames = window.RETRIEVAL[scene];
 selectedVisit = Math.min(selectedVisit, frames.length-1);
 const frame = frames[selectedVisit];
 const title = scene[0].toUpperCase()+scene.slice(1);
 for (const key of ['gt','ours']) {
  const img = document.querySelector('#history-'+key);
  img.src = frame.images[key];
  img.alt = `${title}, visit ${frame.visit}, ${key==='gt'?'held-out observation':'ReScene final recall'}`;
 }
 visits.replaceChildren();
 frames.forEach((f,i)=>{
  const button = document.createElement('button');button.type='button';button.textContent='Visit '+f.visit;
  button.setAttribute('aria-pressed',String(i===selectedVisit));
  button.addEventListener('click',()=>{selectedVisit=i;renderHistory();visits.children[i].focus();});visits.append(button);
 });
 document.querySelector('#history-status').textContent=`${title} at visit ${frame.visit} of ${frames.length}, recovered after the final update.`;
}
sceneSelect.addEventListener('change',renderHistory);renderHistory();
const tabs = [...document.querySelectorAll('[role="tab"]')];
function selectTab(tab) {
 tabs.forEach(t=>{const active=t===tab;t.setAttribute('aria-selected',String(active));t.tabIndex=active?0:-1;document.getElementById(t.getAttribute('aria-controls')).hidden=!active;});
}
tabs.forEach((tab,i)=>{
 tab.addEventListener('click',()=>selectTab(tab));
 tab.addEventListener('keydown',e=>{
  let next;
  if(e.key==='ArrowRight')next=(i+1)%tabs.length;
  if(e.key==='ArrowLeft')next=(i+tabs.length-1)%tabs.length;
  if(e.key==='Home')next=0;
  if(e.key==='End')next=tabs.length-1;
  if(next!==undefined){e.preventDefault();selectTab(tabs[next]);tabs[next].focus();}
 });
});
