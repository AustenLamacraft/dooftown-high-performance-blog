exports.data = {
  title: 'Reveal.js Slideshow',
};

// eslint-disable-next-line func-names
exports.render = function (data) {
  return `<!doctype html>
  <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <link rel="stylesheet" href='${this.url('/js/reveal.js/dist/reveal.css')}'>
      <link rel="stylesheet" href='${this.url('/js/reveal.js/dist/theme/white.css')}'>
      <title>${data.title}</title>
    </head>
    <body>
      <div class="reveal">
        <div class="slides">
          <section data-markdown>
            <textarea data-template>
              ${data.rawMarkdown}
            </textarea>
          </section>
        </div>
      </div>
      <script src='${this.url('/js/reveal.js/dist/reveal.js')}'></script>
      <script src='${this.url('/js/reveal.js/plugin/markdown/markdown.js')}'></script>
      <script src='${this.url('/js/reveal.js/plugin/math/math.js')}'></script>
      <script src='${this.url('/js/reveal.js/plugin/zoom/zoom.js')}'></script>
      <script>
        Reveal.initialize({
          plugins: [ RevealMarkdown, RevealMath.KaTeX, RevealZoom ],
          hash: true
        });
      </script>
    </body>
  </html>`;
};
