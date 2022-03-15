const parseStringPromise = require('xml2js').parseStringPromise;
const EleventyFetch = require("@11ty/eleventy-fetch");


// https://www.mikestreety.co.uk/blog/creating-an-11ty-collection-from-json-api/
module.exports = async function() {
    const apiUrl = 'http://export.arxiv.org/api/query?search_query=au:lamacraft&sortBy=submittedDate&max_results=100'

    return EleventyFetch(apiUrl, {
        duration: "1d", 
        type: "text"    
      }).then(xml => parseStringPromise(xml))
      .then(json => {
          const entry = json.feed.entry
          return entry.map(element => {
            element.author = element.author.map(a => a.name)
            element.published = new Date(element.published)
            return element 
          });
      });
}