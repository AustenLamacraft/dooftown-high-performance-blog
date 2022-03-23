/**
 * Copyright (c) 2020 Google Inc
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 * Copyright (c) 2018 Zach Leatherman
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

const { DateTime } = require("luxon");
const { promisify } = require("util");
const fs = require("fs");
const glob = require("glob");
const path = require("path");
const hasha = require("hasha");
const touch = require("touch");
const readFile = promisify(fs.readFile);
const readdir = promisify(fs.readdir);
const stat = promisify(fs.stat);
const execFile = promisify(require("child_process").execFile);
const pluginRss = require("@11ty/eleventy-plugin-rss");
const pluginSyntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");
const pluginNavigation = require("@11ty/eleventy-navigation");
const markdownIt = require("markdown-it");
const markdownItAnchor = require("markdown-it-anchor");
const localImages = require("./third_party/eleventy-plugin-local-images/.eleventy.js");

const markdownItMathjax3 = require('./third_party/markdown-it-mathjax3/index.js')

const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {CHTML} = require('mathjax-full/js/output/chtml.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AssistiveMmlHandler} = require('mathjax-full/js/a11y/assistive-mml.js');


const CleanCSS = require("clean-css");
const GA_ID = require("./_data/metadata.json").googleAnalyticsId;
const { cspDevMiddleware } = require("./_11ty/apply-csp.js");

const templateFormats = ["md", "njk", "html", "liquid"]

module.exports = function (eleventyConfig) {
  eleventyConfig.addPlugin(pluginRss);
  eleventyConfig.addPlugin(pluginSyntaxHighlight);
  eleventyConfig.addPlugin(pluginNavigation);

  // eleventyConfig.addPlugin(localImages, {
  //   distPath: "_site",
  //   assetPath: "/img/remote",
  //   selector:
  //     "img,amp-img,amp-video,meta[property='og:image'],meta[name='twitter:image'],amp-story",
  //   verbose: false,
  // });

  eleventyConfig.addPlugin(require("./_11ty/img-dim.js"));
  eleventyConfig.addPlugin(require("./_11ty/json-ld.js"));
  eleventyConfig.addPlugin(require("./_11ty/optimize-html.js"));
  eleventyConfig.addPlugin(require("./_11ty/apply-csp.js"));
  eleventyConfig.setDataDeepMerge(true);
  eleventyConfig.addLayoutAlias("post", "layouts/post.njk");
  eleventyConfig.addNunjucksAsyncFilter(
    "addHash",
    function (absolutePath, callback) {
      readFile(path.join(".", absolutePath), {
        encoding: "utf-8",
      })
        .then((content) => {
          return hasha.async(content);
        })
        .then((hash) => {
          callback(null, `${absolutePath}?hash=${hash.substr(0, 10)}`);
        })
        .catch((error) => {
          callback(
            new Error(`Failed to addHash to '${absolutePath}': ${error}`)
          );
        });
    }
  );

  async function lastModifiedDate(filename) {
    try {
      const { stdout } = await execFile("git", [
        "log",
        "-1",
        "--format=%cd",
        filename,
      ]);
      return new Date(stdout);
    } catch (e) {
      console.error(e.message);
      // Fallback to stat if git isn't working.
      const stats = await stat(filename);
      return stats.mtime; // Date
    }
  }
  // Cache the lastModifiedDate call because shelling out to git is expensive.
  // This means the lastModifiedDate will never change per single eleventy invocation.
  const lastModifiedDateCache = new Map();
  eleventyConfig.addNunjucksAsyncFilter(
    "lastModifiedDate",
    function (filename, callback) {
      const call = (result) => {
        result.then((date) => callback(null, date));
        result.catch((error) => callback(error));
      };
      const cached = lastModifiedDateCache.get(filename);
      if (cached) {
        return call(cached);
      }
      const promise = lastModifiedDate(filename);
      lastModifiedDateCache.set(filename, promise);
      call(promise);
    }
  );

  eleventyConfig.addFilter("encodeURIComponent", function (str) {
    return encodeURIComponent(str);
  });

  eleventyConfig.addFilter("cssmin", function (code) {
    return new CleanCSS({}).minify(code).styles;
  });

  eleventyConfig.addFilter("readableDate", (dateObj) => {
    return DateTime.fromJSDate(dateObj, { zone: "utc" }).toFormat(
      "dd LLL yyyy"
    );
  });

  // https://html.spec.whatwg.org/multipage/common-microsyntaxes.html#valid-date-string
  eleventyConfig.addFilter("htmlDateString", (dateObj) => {
    return DateTime.fromJSDate(dateObj, { zone: "utc" }).toFormat("yyyy-LL-dd");
  });

  eleventyConfig.addFilter("sitemapDateTimeString", (dateObj) => {
    const dt = DateTime.fromJSDate(dateObj, { zone: "utc" });
    if (!dt.isValid) {
      return "";
    }
    return dt.toISO();
  });

  // Get the first `n` elements of a collection.
  eleventyConfig.addFilter("head", (array, n) => {
    if (n < 0) {
      return array.slice(n);
    }

    return array.slice(0, n);
  });

  eleventyConfig.addCollection("posts", function (collectionApi) {
    return collectionApi.getFilteredByTag("posts");
  });

  // Pass through revealjs
  // See https://github.com/11ty/eleventy/issues/768#issue-522432961
  eleventyConfig.addPassthroughCopy({ 'node_modules/reveal.js': 'js/reveal.js' });
  // Slides need raw markdown to be processed by reveal
  // https://github.com/11ty/eleventy/issues/1206#issuecomment-718226128
  eleventyConfig.addCollection('talks', (collection) => {
    return (
      collection
        .getFilteredByTag("talks")
        // append the raw content
        .map((item) => {
          item.data.rawMarkdown = item.template.frontMatter.content || '';
          return item;
        })
    );
  });
  

  eleventyConfig.addCollection("recent", (collection) => {
    return (
      collection
        .getAll().concat(collection.getAll()[0].data.arxiv) // Add in publications
        .filter(item => {
          return item.data.tags?.some(tag => ['posts', 'talks', 'arxiv', 'teaching'].includes(tag))
        })
        .sort((a,b) => a.date - b.date)
        .slice(-10)
    )
  })

  eleventyConfig.addCollection("tagList", require("./_11ty/getTagList"));
  eleventyConfig.addPassthroughCopy("img");
  eleventyConfig.addPassthroughCopy("css");
  // We need to copy cached.js only if GA is used
  eleventyConfig.addPassthroughCopy(GA_ID ? "js" : "js/*[!cached].*");
  eleventyConfig.addPassthroughCopy("fonts");

  // We need to rebuild upon JS change to update the CSP.
  eleventyConfig.addWatchTarget("./js/");
  // We need to rebuild on CSS change to inline it.
  eleventyConfig.addWatchTarget("./css/");
  // Unfortunately this means .eleventyignore needs to be maintained redundantly.
  // But without this the JS build artefacts doesn't trigger a build.
  eleventyConfig.setUseGitIgnore(false);

  /* Markdown Overrides */
  let markdownLibrary = markdownIt({
    html: true,
    breaks: true,
    linkify: true,
  }).use(markdownItAnchor, {
    permalink: true,
    permalinkClass: "direct-link",
    permalinkSymbol: "#",
  });
  
  // Add MathJax
  // markdownLibrary.use(markdownItMathjax3, {
  //   loader: {load: ['[tex]/physics', '[tex]/ams', 'output/chtml']},
  //   tex: {
  //     packages: {'[+]': ['physics', 'ams']},
  //     tags: 'all'
  //   },
  // });
  // eleventyConfig.setLibrary("md", markdownLibrary);

  // Process any math on page with MathJax
  // First pass through fonts
  eleventyConfig.addPassthroughCopy({ 'node_modules/mathjax-full/es5/output/chtml/fonts/woff-v2': 'fonts/woff-v2' });
  // Then transform each page
  eleventyConfig.addTransform('mathjax', function(content, outputPath) {
    
    const template = this;

    // Only render posts
    if (!template.inputPath.startsWith('./content/posts/')) { 
      return content;
    }
    
    console.log(`Adding MathJax to ${template.inputPath}`)

    //
    //  Create DOM adaptor and register it for HTML documents
    //
    const adaptor = liteAdaptor();
    AssistiveMmlHandler(RegisterHTMLHandler(adaptor));

    //
    //  Create input and output jax and a document using them on the content from the HTML file
    //
    const tex = new TeX({packages: {'[+]': ['physics', 'ams']}, inlineMath: [['$','$']]});
    const chtml = new CHTML({fontURL: '../../../fonts/woff-v2'});
    const html = mathjax.document(content, {InputJax: tex, OutputJax: chtml});

    //
    //  Typeset the document
    //
    html.render();

    //
    //  If no math was found on the page, remove the stylesheet
    //
    if (Array.from(html.math).length === 0) adaptor.remove(html.outputJax.chtmlStyles);

    //
    //  Output the resulting HTML

    return adaptor.outerHTML(adaptor.root(html.document)).trim()
  })

  // Copy assets to stay with their page`
  // See https://github.com/11ty/eleventy/issues/379#issuecomment-779705668
  eleventyConfig.addTransform("local-images", function(content, outputPath) {
    // HUGO logic:
    // - if multiple *.md are in a folder (ignoring _index.html) - then no asset-copy over will occur
    // - if single index.html (allowing extra _index.html), then no further sub-dirs will be processed, all sub-dirs and files will be copied (except *.md)
    //
    // Alg:
    // - get all md/html/njk in the directory and sub-dirs, ignoring _index.* (_index.* - could be later used to create list-templates)
    // - if only 1 found = we copy the entire sub-content
    // - otherwise do no copy-over nothing
    
    const template = this;

    if (!template.inputPath.startsWith('./content')) { 
      return content;
    }
    // console.warn(`TRANSFORM - input: ${template.inputPath}, output: ${outputPath}`);


    const outputDir = path.dirname(outputPath);       
    const templateDir = path.dirname(template.inputPath).replace(/^\.\//, "");
    const templateFileName = path.basename(template.inputPath);

    const extensionsRegex = templateFormats.join(",");
    const mdSearchPattern = path.join(templateDir, `**/*.{${extensionsRegex}}`);
    const mdIgnorePattern = path.join(templateDir, `**/_index.{${extensionsRegex}}`);

    const entries = glob.sync(mdSearchPattern, { nodir: true, ignore: mdIgnorePattern });
    
    // only 1 page template allowed when copying assets
    if (entries.length > 1) {
        console.info(`Skipping copying over files from: ${templateDir} as multiple templates found in directory!`);
        return content;
    }

    // copy all hierarchically, except templates
    const fileSearchPattern = path.join(templateDir, `**/*`);
    const fileIgnorePattern = path.join(templateDir, `**/*.{${extensionsRegex}}`);

    const filesToCopy = glob.sync(fileSearchPattern, { nodir: true, ignore: fileIgnorePattern });
    for (let filePath of filesToCopy) {
        // strip template dir
        // prepend output dir
        const destPath = path.join(
            outputDir,
            filePath.substring(templateDir.length)
        );

        const destDir = path.dirname(destPath);
         
        fs.mkdirSync(destDir, { recursive: true });
        fs.copyFileSync(filePath, destPath);
    }

    // keep original content
    return content;
  });

  // Browsersync Overrides
  eleventyConfig.setBrowserSyncConfig({
    middleware: cspDevMiddleware,
    callbacks: {
      ready: function (err, browserSync) {
        const content_404 = fs.readFileSync("_site/404.html");

        browserSync.addMiddleware("*", (req, res) => {
          // Provides the 404 content without redirect.
          res.write(content_404);
          res.end();
        });
      },
    },
    ui: false,
    ghostMode: false,
  });

  // Run me before the build starts
  eleventyConfig.on("beforeBuild", () => {
    // Copy _header to dist
    // Don't use addPassthroughCopy to prevent apply-csp from running before the _header file has been copied
    try {
      const headers = fs.readFileSync("./_headers", { encoding: "utf-8" });
      fs.mkdirSync("./_site", { recursive: true });
      fs.writeFileSync("_site/_headers", headers);
    } catch (error) {
      console.log(
        "[beforeBuild] Something went wrong with the _headers file\n",
        error
      );
    }
  });

  // After the build touch any file in the test directory to do a test run.
  eleventyConfig.on("afterBuild", async () => {
    const files = await readdir("test");
    for (const file of files) {
      touch(`test/${file}`);
      break;
    }
  });

  return {
    templateFormats,

    // If your site lives in a different subdirectory, change this.
    // Leading or trailing slashes are all normalized away, so don’t worry about those.

    // If you don’t have a subdirectory, use "" or "/" (they do the same thing)
    // This is only used for link URLs (it does not affect your file structure)
    // Best paired with the `url` filter: https://www.11ty.io/docs/filters/url/

    // You can also pass this in on the command line using `--pathprefix`
    // pathPrefix: "/",

    markdownTemplateEngine: "liquid",
    htmlTemplateEngine: "njk",
    dataTemplateEngine: "njk",

    // These are all optional, defaults are shown:
    dir: {
      input: ".",
      includes: "_includes",
      data: "_data",
      // Warning hardcoded throughout repo. Find and replace is your friend :)
      output: "_site",
    },
  };
};
