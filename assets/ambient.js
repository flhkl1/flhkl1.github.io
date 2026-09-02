/* Site-wide background audio that never stops, plus a persistent on/off toggle.
   Include with:
     <script src="assets/ambient.js" data-src="assets/audio/underground.mp3"
             data-title="Nick Curly - Underground"></script>
   Paths in data-src are relative to the page, so subpages pass ../assets/...

   Two halves:

   1. THE PLAYER. Defaults to on. Browsers refuse to start audible audio before
      the visitor has interacted with the page, so when the first play() is
      rejected we start the track MUTED (always permitted) and unmute on the
      first click / keypress / scroll. The clock is already running by then, so
      sound arrives instantly rather than starting over.

   2. SEAMLESS NAVIGATION. A normal link click tears down the document and kills
      the audio with it. Internal links are therefore intercepted and the next
      page is fetched and swapped into the live document instead. Nothing
      unloads, so the track plays straight through the navigation with no gap
      and no second autoplay prompt. Hard loads (typed URL, refresh, back into a
      cold tab) fall back to resuming from the stored position.

      Because the swap is instant it reads as a cut, so every navigation is
      wrapped in a fade through the site's night blue -- out, swap, back in.
      That covers the portfolio <-> Project 0 <-> 1 <-> any project added later,
      since they all route through here. */
(function(){
  if (window.__ambient) return;          // never initialise twice
  window.__ambient = true;

  var tag   = document.currentScript;
  var SELF  = tag.src;
  var SRC   = new URL(tag.getAttribute('data-src'), document.baseURI).href;
  var TITLE = tag.getAttribute('data-title') || 'Ambient';
  var K_ON  = 'ambient:on';      // "1" / "0"
  var K_POS = 'ambient:pos';     // seconds into the track
  var K_AT  = 'ambient:at';      // epoch ms of the last position write
  var START_AT = 88;             // 1:28 -- where the track is asked to open

  function get(k){ try{ return localStorage.getItem(k); }catch(e){ return null; } }
  function set(k,v){ try{ localStorage.setItem(k,v); }catch(e){} }
  function wanted(){ return get(K_ON) !== '0'; }

  /* ---------------------------------------------------------------- player */

  var audio = new Audio(SRC);
  audio.loop = true;
  audio.preload = 'auto';
  audio.volume = 0.45;

  // Open at START_AT. If a previous page left a position behind, pick that up
  // instead -- plus the time spent loading this one -- so moving between pages
  // is continuous rather than jumping back to 1:28 on every navigation.
  audio.addEventListener('loadedmetadata', function(){
    var pos = parseFloat(get(K_POS) || '0');
    var at  = parseFloat(get(K_AT)  || '0');
    if (!isFinite(pos) || pos <= 0) {
      pos = START_AT;                       // nothing stored: start at 1:28
    } else if (at) {
      pos += Math.min((Date.now() - at) / 1000, 30);
    }
    if (audio.duration) {
      if (pos >= audio.duration) pos = pos % audio.duration;
      if (pos >= audio.duration) pos = 0;   // START_AT past the end of a short file
    }
    try { audio.currentTime = pos; } catch(e){}
  });

  var armed = false;
  // click / pointerdown / keydown / touchend grant user activation; scroll and
  // mousemove do not, but they are cheap to listen for and cost nothing if the
  // retry fails, because onGesture re-arms on rejection.
  var GESTURES = ['pointerdown','pointerup','click','keydown','touchstart','touchend','wheel','scroll'];

  function audible(){ return !audio.paused && !audio.muted; }

  function start(){
    audio.muted = false;
    return audio.play().catch(function(){
      // Audible autoplay refused. Run silently so the track is already in
      // position, and wait for the gesture that lets us turn it up.
      audio.muted = true;
      audio.play().catch(function(){});
      arm();
    });
  }
  function arm(){
    if (armed) return;
    armed = true;
    if (btn) paint();   // btn does not exist on the first arm()
    GESTURES.forEach(function(ev){
      window.addEventListener(ev, onGesture, {once:true, passive:true});
    });
  }
  function disarm(){
    if (!armed) return;
    armed = false;
    if (btn) paint();   // btn does not exist on the first arm()
    GESTURES.forEach(function(ev){ window.removeEventListener(ev, onGesture); });
  }
  function onGesture(){
    disarm();
    if (!wanted()) return;
    audio.muted = false;
    audio.play().catch(function(){
      // That event fired but did not count as activation (a scroll or a wheel).
      // Drop back to silent playback and keep waiting for a real one.
      audio.muted = true;
      audio.play().catch(function(){});
      arm();
    });
  }

  // Try the moment the script runs, before the page has even finished parsing.
  if (wanted()) start();

  function remember(){
    if (!audio.paused) { set(K_POS, String(audio.currentTime)); set(K_AT, String(Date.now())); }
  }
  setInterval(remember, 4000);
  window.addEventListener('pagehide', remember);
  document.addEventListener('visibilitychange', function(){ if (document.hidden) remember(); });

  /* --------------------------------------------------------------- control */

  var css = document.createElement('style');
  css.setAttribute('data-ambient','');   // survives the page-swap head cleanup
  css.textContent = [
    '.ambient{position:fixed;right:22px;bottom:20px;z-index:50;display:flex;',
    '  align-items:center;gap:10px;padding:9px 15px;cursor:pointer;',
    '  background:rgba(9,17,42,.62);border:1px solid rgba(120,160,220,.22);',
    '  border-radius:2px;backdrop-filter:blur(4px);color:#fff;',
    '  font-family:"Michroma","Futura",sans-serif;font-size:9px;letter-spacing:.22em;',
    '  text-transform:uppercase;line-height:1;',
    '  transition:border-color .22s ease,opacity .22s ease;opacity:.72}',
    '.ambient:hover{opacity:1;border-color:rgba(255,255,255,.45)}',
    '.ambient .eq{display:flex;align-items:flex-end;gap:2px;height:11px;width:13px}',
    '.ambient .eq i{width:2px;background:#fff;height:3px;transform-origin:bottom}',
    '.ambient.on .eq i{animation:ambient-eq .9s ease-in-out infinite}',
    '.ambient.on .eq i:nth-child(2){animation-delay:.18s}',
    '.ambient.on .eq i:nth-child(3){animation-delay:.36s}',
    '.ambient.on .eq i:nth-child(4){animation-delay:.09s}',
    '.ambient.off .eq i{height:3px;opacity:.5}',
    '.ambient.pending{opacity:1;border-color:rgba(255,255,255,.55)}',
    '.ambient.pending .eq i{animation:ambient-wait 1.6s ease-in-out infinite}',
    '.ambient.pending .eq i:nth-child(2){animation-delay:.12s}',
    '.ambient.pending .eq i:nth-child(3){animation-delay:.24s}',
    '.ambient.pending .eq i:nth-child(4){animation-delay:.36s}',
    '@keyframes ambient-wait{0%,100%{opacity:.35}50%{opacity:1}}',
    '@keyframes ambient-eq{0%,100%{height:3px}50%{height:11px}}',
    '@media(prefers-reduced-motion:reduce){.ambient.on .eq i{animation:none;height:7px}}',
    '@media(max-width:640px){.ambient{right:14px;bottom:14px;padding:8px 12px;font-size:8px}}'
  ].join('');
  document.head.appendChild(css);

  var btn = document.createElement('button');
  btn.className = 'ambient off';
  btn.type = 'button';
  btn.title = TITLE;
  btn.innerHTML = '<span class="eq"><i></i><i></i><i></i><i></i></span><span class="lbl"></span>';
  var lbl = btn.querySelector('.lbl');

  function paint(){
    var on = audible();
    var pending = !on && armed && wanted();   // blocked by autoplay, not by choice
    btn.classList.toggle('on', on);
    btn.classList.toggle('off', !on);
    btn.classList.toggle('pending', pending);
    lbl.textContent = on ? 'Sound On' : (pending ? 'Play Sound' : 'Sound Off');
    btn.setAttribute('aria-label', (on ? 'Mute' : 'Play') + ' background music: ' + TITLE);
    btn.setAttribute('aria-pressed', String(on));
  }
  paint();
  ['play','pause','volumechange'].forEach(function(ev){ audio.addEventListener(ev, paint); });

  btn.addEventListener('click', function(e){
    e.stopPropagation();
    if (audible()) { audio.pause(); set(K_ON,'0'); }
    else { set(K_ON,'1'); disarm(); start(); }
  });

  /* ----------------------------------------------------------------- fade */

  /* The veil hangs off <html> rather than <body>: the swap below replaces the
     whole of body, and anything living in there would be torn out mid-fade. */
  var FADE_OUT = 320, FADE_IN = 440;
  var reduce = window.matchMedia('(prefers-reduced-motion:reduce)');

  var veil = document.createElement('div');
  veil.setAttribute('data-ambient-veil','');
  veil.style.cssText =
    'position:fixed;inset:0;z-index:9998;pointer-events:none;background:#060a1a;' +
    'opacity:0;transition:opacity ' + FADE_OUT + 'ms cubic-bezier(.22,.61,.36,1)';
  document.documentElement.appendChild(veil);

  function fade(to, ms){
    return new Promise(function(done){
      if (reduce.matches){ veil.style.opacity = '0'; return done(); }
      veil.style.transitionDuration = ms + 'ms';
      /* opaque veil also swallows clicks, so a second link can't be hit
         while the first navigation is still in the air */
      veil.style.pointerEvents = to ? 'auto' : 'none';
      void veil.offsetWidth;              // flush, so 0 -> 1 in one tick animates
      veil.style.opacity = to ? '1' : '0';
      setTimeout(done, ms);
    });
  }

  /* ------------------------------------------------------------ navigation */

  function internal(a){
    if (!a || a.target || a.hasAttribute('download')) return null;
    var u;
    try { u = new URL(a.href, document.baseURI); } catch(e){ return null; }
    if (u.origin !== location.origin) return null;
    if (!/(\/|\.html)$/.test(u.pathname)) return null;
    if (u.pathname === location.pathname) return null;   // same page: let the hash work
    return u;
  }

  // Update the URL *before* swapping: relative paths inside the incoming markup
  // and stylesheets resolve against the document URL, and the two pages sit at
  // different depths (/index.html vs /0/index.html).
  function render(doc, url){
    document.title = doc.title;

    var old = document.head.querySelectorAll('style:not([data-ambient])');
    Array.prototype.forEach.call(old, function(n){ n.remove(); });
    Array.prototype.forEach.call(doc.head.querySelectorAll('style'), function(n){
      document.head.appendChild(n.cloneNode(true));
    });

    document.body.innerHTML = doc.body.innerHTML;
    document.body.className = doc.body.className;
    document.body.appendChild(btn);

    // innerHTML leaves scripts inert; swap each for an executable copy in place.
    Array.prototype.forEach.call(document.body.querySelectorAll('script'), function(s){
      var src = s.getAttribute('src');
      if (src && new URL(src, document.baseURI).href === SELF) { s.remove(); return; }
      var n = document.createElement('script');
      Array.prototype.forEach.call(s.attributes, function(a){ n.setAttribute(a.name, a.value); });
      if (!src) n.textContent = s.textContent;
      s.replaceWith(n);
    });

    var target = url.hash && document.querySelector(url.hash);
    if (target) target.scrollIntoView();
    else window.scrollTo(0, 0);
  }

  /* Fade out and fetch at the same time, so the wait costs no extra time: by
     the time the veil is down the next page is usually already in hand. */
  function go(url, push){
    var page = fetch(url.href, {credentials:'same-origin'})
      .then(function(r){ if (!r.ok) throw 0; return r.text(); });

    return Promise.all([fade(1, FADE_OUT), page])
      .then(function(both){
        if (push) history.pushState({ambient:1}, '', url.href);
        render(new DOMParser().parseFromString(both[1], 'text/html'), url);
        return fade(0, FADE_IN);
      })
      .catch(function(){ location.href = url.href; });   // fall back to a real load
  }

  // Exposed so the avatar transition can hand over once its video finishes:
  // a hard location change would tear the document down and cut the music.
  window.__ambientNav = function(href){
    var u; try { u = new URL(href, document.baseURI); } catch(e){ location.href = href; return; }
    if (u.origin !== location.origin) { location.href = u.href; return; }
    go(u, true);
  };

  document.addEventListener('click', function(e){
    if (e.defaultPrevented || e.button !== 0) return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    var url = internal(e.target.closest && e.target.closest('a[href]'));
    if (!url) return;
    e.preventDefault();
    go(url, true);
  });

  window.addEventListener('popstate', function(){
    go(new URL(location.href), false);
  });

  function mount(){ document.body.appendChild(btn); paint(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount);
  else mount();
})();
