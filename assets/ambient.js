/* Site-wide background audio, kept as a secret: the site is silent until the
   visitor presses M, and M again turns it back off. There is no visible control.
   Include with:
     <script src="assets/ambient.js" data-src="assets/audio/underground.mp3"
             data-title="Nick Curly - Underground"></script>
   Paths in data-src are relative to the page, so subpages pass ../assets/...

   Two halves:

   1. THE PLAYER. Off by default, and nothing but M ever starts it -- not
      opening a project, not a click, not a reload. Pressing M is itself a user
      gesture, so the track starts audibly straight away.

   2. SEAMLESS NAVIGATION. A normal link click tears down the document and kills
      the audio with it. Internal links are therefore intercepted and the next
      page is fetched and swapped into the live document instead. Nothing
      unloads, so the track plays straight through the navigation with no gap
      and no second autoplay prompt.

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
  var START_AT = 0;              // seconds -- where the track opens

  /* ---------------------------------------------------------------- player */

  var audio = new Audio(SRC);
  audio.loop = true;
  audio.preload = 'none';        // nothing to fetch until someone asks for it
  audio.volume = 0.45;

  audio.addEventListener('loadedmetadata', function(){
    var pos = START_AT;
    if (audio.duration && pos >= audio.duration) pos = 0;
    try { audio.currentTime = pos; } catch(e){}
  });

  document.addEventListener('keydown', function(e){
    if (e.key !== 'm' && e.key !== 'M') return;
    if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
    var t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    if (audio.paused) audio.play().catch(function(){});
    else audio.pause();
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
})();
