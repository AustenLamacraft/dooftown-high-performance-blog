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
          const entries = json.feed.entry
          const data = ({id, published, author, title, summary, ...rest}) => ({
              ...rest, 
              author: author.map(a => a.name),
              title: title[0],
              summary: summary[0],
              tags: ["arxiv"]
            })
          return entries
            .map(entry => ({
                data: data(entry),
                url: entry.id[0],
                date: new Date(entry.published)
            }))
            .sort((a,b) => a.date - b.date)
      });
}