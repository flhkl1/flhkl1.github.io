/* Project write-ups as plain Markdown.

   Each project folder is just:
     N/index.html   a tiny shell (never needs editing)
     N/post.md      the whole write-up: text, images, math, widgets

   This script fetches post.md, renders it with marked + KaTeX (vendored under
   assets/vendor, so nothing depends on a CDN), then wires up the few
   interactive pieces. ambient.js re-runs this script on every in-place
   navigation, so it loads its own dependencies instead of relying on <script>
   order, and it only ever works on the #post element that is on the page now.

   Markdown conventions on top of normal Markdown:

   - A paragraph that is only images becomes a row of figures. The image
     title is the caption:   ![alt](src "caption")  ![alt](src "caption")
   - A paragraph starting with TODO is a dashed placeholder. Anything up to
     " — " is its label. Placeholders are hidden when printing.
   - A paragraph starting with NEEDED is an orange "result still missing" box.
   - Math: $inline$ and $$display$$ (display math on its own lines).
   - Widgets are one-line HTML tags; see README-posts.md for each one. */
(function () {
  var art = document.getElementById('post');
  if (!art) return;

  var SELF = document.currentScript && document.currentScript.src;
  var BASE = SELF ? SELF.replace(/post\.js(\?.*)?$/, '') : '../assets/';

  function load(path) {
    return new Promise(function (ok, bad) {
      var s = document.createElement('script');
      s.src = BASE + path;
      s.onload = ok;
      s.onerror = bad;
      document.head.appendChild(s);
    });
  }
  function stylesheet(path) {
    var href = BASE + path;
    if (document.querySelector('link[href="' + href + '"]')) return;
    var l = document.createElement('link');
    l.rel = 'stylesheet';
    l.href = href;
    document.head.appendChild(l);
  }

  stylesheet('vendor/katex/katex.min.css');
  var deps = [
    window.marked ? Promise.resolve() : load('vendor/marked.umd.js'),
    window.katex ? Promise.resolve() : load('vendor/katex/katex.min.js').catch(function () {})
  ];
  var src = art.getAttribute('data-src') || 'post.md';
  var text = fetch(src, { cache: 'no-cache' }).then(function (r) {
    if (!r.ok) throw new Error(r.status);
    return r.text();
  });

  Promise.all([text].concat(deps)).then(function (v) {
    render(v[0]);
  }).catch(function (e) {
    art.innerHTML = '<p class="todo">Could not load <code>' + src + '</code> (' + e +
      '). If you opened this file directly, serve the folder instead: <code>python3 -m http.server</code></p>';
  });

  /* ------------------------------------------------------------- helpers */

  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function slug(s) {
    return String(s).toLowerCase().replace(/<[^>]+>/g, '').replace(/&[a-z]+;/g, '')
      .replace(/[^\wÀ-￿]+/g, '-').replace(/^-+|-+$/g, '');
  }
  function frontMatter(md) {
    var meta = {};
    var m = md.match(/^---\s*\n([\s\S]*?)\n---\s*\n/);
    if (!m) return { meta: meta, body: md };
    m[1].split('\n').forEach(function (line) {
      var i = line.indexOf(':');
      if (i > 0) meta[line.slice(0, i).trim()] = line.slice(i + 1).trim();
    });
    return { meta: meta, body: md.slice(m[0].length) };
  }
  function tex(src, display) {
    if (window.katex) {
      try { return katex.renderToString(src, { displayMode: display, throwOnError: false }); }
      catch (e) {}
    }
    return '<code class="tex">' + esc(src) + '</code>';
  }

  /* ------------------------------------------------------------- render */

  function render(md) {
    var fm = frontMatter(md.replace(/\r\n/g, '\n'));
    var meta = fm.meta, body = fm.body;

    // Keep code away from the math scanner, then pull the math out so marked
    // never sees backslashes or underscores inside it.
    var code = [], math = [];
    body = body.replace(/```[\s\S]*?```|`[^`\n]+`/g, function (m) {
      code.push(m); return 'C' + (code.length - 1) + '';
    });
    body = body.replace(/^[ \t]*\$\$([\s\S]+?)\$\$[ \t]*$/gm, function (m, t) {
      math.push([t, true]); return '\n<div class="math">M' + (math.length - 1) + '</div>\n';
    });
    body = body.replace(/(^|[^\\$\w])\$([^\s$](?:[^$\n]*?[^\s$\\])?)\$(?![\w$])/g, function (m, pre, t) {
      math.push([t, false]); return pre + 'M' + (math.length - 1) + '';
    });
    body = body.replace(/C(\d+)/g, function (m, i) { return code[+i]; });

    var renderer = {
      heading: function (t) {
        var inner = this.parser.parseInline(t.tokens);
        return '<h' + t.depth + ' id="' + slug(t.text) + '">' + inner + '</h' + t.depth + '>\n';
      },
      paragraph: function (t) {
        var toks = t.tokens || [];
        var imgs = toks.filter(function (x) { return x.type === 'image'; });
        var onlyImages = imgs.length && toks.every(function (x) {
          return x.type === 'image' || x.type === 'br' || (x.type === 'text' && !x.raw.trim());
        });
        if (onlyImages) {
          return '<div class="figs n' + imgs.length + '">' + imgs.map(function (im) {
            return '<figure><img src="' + esc(im.href) + '" alt="' + esc(im.text) + '" loading="lazy">' +
              (im.title ? '<figcaption>' + md2.parseInline(im.title) + '</figcaption>' : '') +
              '</figure>';
          }).join('') + '</div>\n';
        }
        var html = this.parser.parseInline(toks);
        var m = t.text.match(/^(TODO|NEEDED)\b/);
        if (m) {
          var rest = html.replace(/^(TODO|NEEDED)\b\s*/, '');
          var cut = rest.indexOf(' — ');
          var label = m[1] + (cut > -1 ? ' ' + rest.slice(0, cut) : '');
          if (cut > -1) rest = rest.slice(cut + 3);
          return '<div class="' + m[1].toLowerCase() + '"><span class="label">' + label + '</span>' + rest + '</div>\n';
        }
        return '<p>' + html + '</p>\n';
      }
    };
    var md2 = new marked.Marked({ gfm: true });
    md2.use({ renderer: renderer });

    var html = md2.parse(body);
    html = html.replace(/M(\d+)/g, function (m, i) { return tex(math[+i][0], math[+i][1]); });

    var head = (meta.cover ? '<img class="cover" src="' + esc(meta.cover) + '" alt="">' : '') +
      '<header class="post-head">' +
      '<a class="home" href="../">← Jack Yang · CS 180</a>' +
      '<h1 id="title">' + esc(meta.title || document.title) + '</h1>' +
      '<p class="byline">' + [meta.subtitle, meta.date].filter(Boolean).map(esc).join(' · ') + '</p>' +
      '</header>';
    art.innerHTML = head + '<div class="post-body">' + html + '</div>' +
      '<footer class="post-foot"><a href="../">← Back to all projects</a></footer>';
    if (meta.title) document.title = meta.title + ' · Jack Yang';

    widgets(art);
    if (meta.effect === 'flow') flowTitle(document.getElementById('title'));
    if (meta.effect === 'rgb') rgbPlates(art.querySelector('img.cover'), document.getElementById('title'));
    if (location.hash) {
      var target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
      if (target) target.scrollIntoView();
    }
  }

  /* ------------------------------------------------------------- widgets */

  function pairs(s) {            // "A=x|B=y" -> [[A,x],[B,y]]
    return (s || '').split('|').filter(Boolean).map(function (p) {
      var i = p.indexOf('=');
      return i < 0 ? [p, p] : [p.slice(0, i).trim(), p.slice(i + 1).trim()];
    });
  }
  function chips(el, list, onPick) {
    if (list.length < 2) return;
    var wrap = document.createElement('div');
    wrap.className = 'chips';
    list.forEach(function (it, i) {
      var b = document.createElement('button');
      b.type = 'button';
      b.textContent = it[0];
      b.addEventListener('click', function () {
        wrap.querySelectorAll('button').forEach(function (x) { x.classList.toggle('on', x === b); });
        onPick(it, i);
      });
      if (i === 0) b.classList.add('on');
      wrap.appendChild(b);
    });
    el.appendChild(wrap);
  }
  function caption(el) {
    var c = el.getAttribute('data-caption');
    if (!c) return;
    var f = document.createElement('p');
    f.className = 'cap';
    f.innerHTML = c;
    el.appendChild(f);
  }

  function widgets(root) {
    /* slider: one or more images driven by a single range input */
    root.querySelectorAll('.slider').forEach(function (el) {
      var tpl = el.getAttribute('data-src');
      var vals = el.getAttribute('data-values').split(',').map(function (s) { return s.trim(); });
      var labels = (el.getAttribute('data-labels') || '').split(',').map(function (s) { return s.trim(); });
      var sets = pairs(el.getAttribute('data-sets'));
      var panels = (el.getAttribute('data-panels') || '').split(',').filter(Boolean);
      var plabels = (el.getAttribute('data-panel-labels') || '').split(',');
      var idx = +(el.getAttribute('data-start') || 0);
      var set = sets.length ? sets[0][1] : '';
      var label = function (i) { return labels[i] || vals[i]; };
      var url = function (v, p) { return tpl.replace(/\{set\}/g, set).replace(/\{v\}/g, v).replace(/\{p\}/g, p || ''); };

      el.innerHTML = '';
      chips(el, sets, function (it) { set = it[1]; thumbs(); show(idx); });
      var stage = document.createElement('div');
      stage.className = 'stage' + (panels.length > 1 ? ' panels n' + panels.length : '');
      var imgs = (panels.length ? panels : ['']).map(function (p, k) {
        var f = document.createElement('figure');
        var im = document.createElement('img');
        im.className = 'zoom';
        f.appendChild(im);
        if (panels.length > 1) {
          var c = document.createElement('figcaption');
          c.textContent = plabels[k] || p;
          f.appendChild(c);
        }
        stage.appendChild(f);
        return im;
      });
      el.appendChild(stage);
      var ctl = document.createElement('div');
      ctl.className = 'ctl';
      ctl.innerHTML = '<input type="range" min="0" step="1"><span class="readout"></span>';
      el.appendChild(ctl);
      var range = ctl.querySelector('input'), out = ctl.querySelector('.readout');
      range.max = vals.length - 1;
      range.setAttribute('aria-label', el.getAttribute('data-caption') || 'step');
      var strip = null, btns = [];
      if (!panels.length || panels.length === 1) {
        strip = document.createElement('div');
        strip.className = 'thumbs';
        el.appendChild(strip);
      }
      function thumbs() {
        if (!strip) return;
        strip.innerHTML = '';
        btns = vals.map(function (v, i) {
          var b = document.createElement('button');
          b.type = 'button';
          b.title = label(i);
          b.innerHTML = '<img loading="lazy" alt="" src="' + esc(url(v, panels[0])) + '">';
          b.addEventListener('click', function () { show(i); });
          strip.appendChild(b);
          return b;
        });
      }
      function show(i) {
        idx = i;
        imgs.forEach(function (im, k) { im.src = url(vals[i], panels[k]); im.alt = label(i); });
        range.value = i;
        out.textContent = label(i);
        btns.forEach(function (b, j) { b.classList.toggle('on', j === i); });
      }
      range.addEventListener('input', function () { show(+range.value); });
      caption(el);
      thumbs();
      show(Math.min(idx, vals.length - 1));
    });

    /* compare: drag to reveal one image over another */
    root.querySelectorAll('.compare').forEach(function (el) {
      var sets = pairs(el.getAttribute('data-sets'));
      var set = sets.length ? sets[0][1] : '';
      var L = el.getAttribute('data-left'), R = el.getAttribute('data-right');
      var url = function (t) { return t.replace(/\{set\}/g, set); };
      el.innerHTML = '';
      chips(el, sets, function (it) { set = it[1]; a.src = url(R); b.src = url(L); });
      var box = document.createElement('div');
      box.className = 'cmp';
      box.innerHTML = '<img class="base" alt=""><div class="top"><img alt=""></div><div class="handle"></div>' +
        '<span class="tag l"></span><span class="tag r"></span><input type="range" min="0" max="100" value="50">';
      el.appendChild(box);
      var a = box.querySelector('.base'), b = box.querySelector('.top img');
      a.src = url(R); b.src = url(L);
      box.querySelector('.tag.l').textContent = el.getAttribute('data-left-label') || '';
      box.querySelector('.tag.r').textContent = el.getAttribute('data-right-label') || '';
      var input = box.querySelector('input');
      input.setAttribute('aria-label', 'Drag to compare');
      input.addEventListener('input', function () { box.style.setProperty('--pos', input.value + '%'); });
      caption(el);
    });

    /* distance: shrink an image to simulate stepping back */
    root.querySelectorAll('.distance').forEach(function (el) {
      var list = pairs(el.getAttribute('data-images'));
      el.innerHTML = '';
      chips(el, list, function (it) { im.src = it[1]; im.alt = it[0]; });
      var stage = document.createElement('div');
      stage.className = 'stage';
      var im = document.createElement('img');
      im.src = list[0][1]; im.alt = list[0][0];
      stage.appendChild(im);
      el.appendChild(stage);
      var ctl = document.createElement('div');
      ctl.className = 'ctl';
      ctl.innerHTML = '<span class="readout">far</span><input type="range" min="5" max="100" value="100" aria-label="Image size"><span class="readout">close</span>';
      el.appendChild(ctl);
      var r = ctl.querySelector('input');
      r.addEventListener('input', function () { im.style.width = r.value + '%'; });
      caption(el);
    });

    /* flip: a visual anagram you can turn over */
    root.querySelectorAll('.flip').forEach(function (el) {
      var up = el.getAttribute('data-up'), down = el.getAttribute('data-down');
      el.innerHTML = '<figure><img class="zoom" alt="' + esc(up + ' / ' + down) + '" src="' +
        esc(el.getAttribute('data-src')) + '"></figure><p class="cap"></p><button type="button"></button>';
      var cap = el.querySelector('.cap'), btn = el.querySelector('button');
      function sync() {
        var r = el.classList.contains('flipped');
        cap.innerHTML = (r ? 'rotated 180°: ' : 'upright: ') + esc(r ? down : up);
        btn.textContent = r ? '↺ Rotate back' : '↻ Rotate 180°';
      }
      btn.addEventListener('click', function () { el.classList.toggle('flipped'); sync(); });
      sync();
    });

    /* lightbox for any figure image */
    var box = document.getElementById('lightbox');
    if (!box) {
      box = document.createElement('div');
      box.id = 'lightbox';
      box.hidden = true;
      box.tabIndex = -1;
      box.innerHTML = '<img alt="">';
      document.body.appendChild(box);
    }
    box.onclick = function () { box.hidden = true; };
    box.onkeydown = function (e) { if (e.key === 'Escape') box.hidden = true; };
    root.addEventListener('click', function (e) {
      var im = e.target.closest && e.target.closest('.figs img, img.zoom');
      if (!im) return;
      box.querySelector('img').src = im.src;
      box.hidden = false;
      box.focus();
    });
  }

  /* ------------------------------------------------------------- rgb plates
     Project 1 only: the cover is split into its R, G and B channels (an SVG
     colour matrix each) and recombined with screen blending, so aligned it is
     the plain image. B is the reference, as in the project. It arrives
     misaligned and snaps into place coarse to fine, halving the offsets each
     step like the pyramid; the cursor pulls the plates apart and leaving
     re-aligns them; a click scrambles them again. The title carries the same
     offsets, scaled down, as cyan/magenta/yellow copies multiplied together. */
  function rgbPlates(cover, h1) {
    if (!cover || matchMedia('(prefers-reduced-motion:reduce)').matches) return;

    if (!document.getElementById('plate-r')) {
      var rows = { r: '1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0',
                   g: '0 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0',
                   b: '0 0 0 0 0  0 0 0 0 0  0 0 1 0 0  0 0 0 1 0' };
      var defs = Object.keys(rows).map(function (c) {
        return '<filter id="plate-' + c + '" color-interpolation-filters="sRGB">' +
               '<feColorMatrix type="matrix" values="' + rows[c] + '"/></filter>';
      }).join('');
      var svg = document.createElement('div');
      svg.innerHTML = '<svg width="0" height="0" style="position:absolute" aria-hidden="true">' + defs + '</svg>';
      document.body.appendChild(svg.firstChild);
    }

    var wrap = document.createElement('div');
    wrap.className = 'cover rgb-cover';
    wrap.title = 'Move to misalign the plates · click to scramble';
    cover.parentNode.replaceChild(wrap, cover);
    cover.className = 'plain';                        // kept for print
    wrap.appendChild(cover);
    var plates = ['b', 'g', 'r'].map(function (c) {
      var im = document.createElement('img');
      im.src = cover.src; im.alt = ''; im.setAttribute('aria-hidden', 'true');
      im.style.filter = 'url(#plate-' + c + ')';
      wrap.appendChild(im);
      return im;
    });
    var ro = document.createElement('span');
    ro.className = 'ro';
    wrap.appendChild(ro);

    if (h1) {
      var text = h1.textContent;
      h1.setAttribute('data-text', text);
      h1.innerHTML = '<span class="txt">' + esc(text) + '</span>';
      h1.classList.add('rgb');
    }

    var MAX = 28, TITLE = 0.15;
    var cur = { gx: 0, gy: 0, rx: 0, ry: 0 }, tgt = { gx: 0, gy: 0, rx: 0, ry: 0 };
    var raf = 0, stepper = 0, hold = 0;

    function apply() {
      var s = 1 + (2 * MAX + 4) / Math.max(1, wrap.clientHeight);   // hide the edges
      plates[1].style.transform = 'translate(' + cur.gx + 'px,' + cur.gy + 'px) scale(' + s + ')';
      plates[2].style.transform = 'translate(' + cur.rx + 'px,' + cur.ry + 'px) scale(' + s + ')';
      plates[0].style.transform = 'scale(' + s + ')';
      var r = function (v) { return Math.round(v); };
      ro.textContent = 'G (' + r(cur.gy) + ', ' + r(cur.gx) + ')  ·  R (' + r(cur.ry) + ', ' + r(cur.rx) + ')' +
        (r(cur.gx) || r(cur.gy) || r(cur.rx) || r(cur.ry) ? '' : '  ·  aligned');
      if (h1) {
        h1.style.setProperty('--gx', cur.gx * TITLE + 'px');
        h1.style.setProperty('--gy', cur.gy * TITLE + 'px');
        h1.style.setProperty('--rx', cur.rx * TITLE + 'px');
        h1.style.setProperty('--ry', cur.ry * TITLE + 'px');
      }
    }

    function stop() { cancelAnimationFrame(raf); clearTimeout(stepper); clearTimeout(hold); raf = stepper = hold = 0; }

    // Coarse to fine: halve every offset per step, then a last snap to zero.
    function align() {
      stop();
      (function step() {
        var done = true;
        for (var k in cur) {
          cur[k] = Math.abs(cur[k]) <= 1 ? 0 : Math.trunc(cur[k] / 2);
          if (cur[k]) done = false;
        }
        apply();
        if (!done) stepper = setTimeout(step, 150);
      })();
    }

    // A fresh, raw-plate-looking misalignment: mostly vertical, R about twice G.
    function scramble() {
      stop();
      var sx = Math.random() < .5 ? -1 : 1, sy = Math.random() < .5 ? -1 : 1;
      cur.gy = sy * (8 + Math.random() * 6);  cur.gx = sx * Math.random() * 6;
      cur.ry = cur.gy * 2 + (Math.random() - .5) * 6;
      cur.rx = cur.gx * 2 + (Math.random() - .5) * 6;
      apply();
      hold = setTimeout(align, 650);
    }

    function follow() {
      var moving = false;
      for (var k in cur) {
        cur[k] += (tgt[k] - cur[k]) * 0.18;
        if (Math.abs(tgt[k] - cur[k]) > 0.05) moving = true;
      }
      apply();
      raf = moving ? requestAnimationFrame(follow) : 0;
    }

    wrap.addEventListener('pointermove', function (e) {
      var b = wrap.getBoundingClientRect();
      var nx = ((e.clientX - b.left) / b.width) * 2 - 1, ny = ((e.clientY - b.top) / b.height) * 2 - 1;
      tgt.gx = nx * MAX / 2; tgt.gy = ny * MAX / 2;
      tgt.rx = nx * MAX;     tgt.ry = ny * MAX;
      clearTimeout(stepper); clearTimeout(hold); stepper = hold = 0;
      if (!raf) raf = requestAnimationFrame(follow);
    });
    wrap.addEventListener('pointerleave', function () {
      for (var k in cur) cur[k] = Math.round(cur[k]);
      align();
    });
    wrap.addEventListener('click', scramble);
    addEventListener('resize', apply);

    if (cover.complete) scramble(); else cover.addEventListener('load', scramble);
  }

  /* ------------------------------------------------------------- flow title
     Project 3 only: the title arrives as a sample. It is rasterised onto a
     coarse grid as x1, each cell gets its own Gaussian eps, and the canvas
     shows x_t = (1 - t) eps + t x1 while t runs 0 -> 1. Click to resample. */
  function flowTitle(h1) {
    if (!h1 || matchMedia('(prefers-reduced-motion:reduce)').matches) return;
    var text = h1.textContent;
    h1.innerHTML = '<span class="txt">' + esc(text) + '</span>';
    h1.classList.add('flow');
    var span = h1.firstChild;
    var cv = document.createElement('canvas');
    cv.setAttribute('aria-hidden', 'true');
    h1.appendChild(cv);
    var ctx = cv.getContext('2d'), raf = 0;

    function gauss() {
      var u = 1 - Math.random(), v = Math.random();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    function run() {
      cancelAnimationFrame(raf);
      var box = span.getBoundingClientRect(), hb = h1.getBoundingClientRect();
      if (!box.width) return;
      var cell = Math.max(2, Math.round(parseFloat(getComputedStyle(h1).fontSize) / 12));
      var cols = Math.ceil(box.width / cell) + 2, rows = Math.ceil(box.height / cell) + 2;
      cv.style.left = (box.left - hb.left - cell) + 'px';
      cv.style.top = (box.top - hb.top - cell) + 'px';
      cv.style.width = cols * cell + 'px';
      cv.style.height = rows * cell + 'px';
      cv.width = cols; cv.height = rows;
      var cs = getComputedStyle(h1);
      ctx.clearRect(0, 0, cols, rows);
      ctx.font = cs.fontWeight + ' ' + parseFloat(cs.fontSize) / cell + 'px ' + cs.fontFamily;
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#000';
      ctx.fillText(text, 1, rows / 2);
      var a = ctx.getImageData(0, 0, cols, rows).data, n = cols * rows;
      var x1 = new Float32Array(n), eps = new Float32Array(n);
      for (var i = 0; i < n; i++) { x1[i] = a[i * 4 + 3] / 255; eps[i] = gauss(); }
      var img = ctx.createImageData(cols, rows), d = img.data;
      var steps = 50, k = 0, last = 0;
      h1.classList.add('flowing');
      cv.style.opacity = 1;
      (function tick(now) {
        if (now - last > 30) {
          last = now;
          var t = Math.min(1, k / steps);
          for (var i = 0; i < n; i++) {
            var v = (1 - t) * (0.45 + 0.3 * eps[i]) + t * x1[i];
            v = v < 0 ? 0 : v > 1 ? 1 : v;
            d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = 34;
            d[i * 4 + 3] = v * 255;
          }
          ctx.putImageData(img, 0, 0);
          k++;
        }
        if (k <= steps) raf = requestAnimationFrame(tick);
        else { cv.style.opacity = 0; h1.classList.remove('flowing'); }
      })(0);
    }
    var go = function () { setTimeout(run, 200); };
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(go); else go();
    h1.addEventListener('click', run);
    h1.title = 'Click to sample again';
  }
})();
