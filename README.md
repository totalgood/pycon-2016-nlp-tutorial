# Making Connections with Natural Language Processing

## Description

Have you ever dreamed of using python to teach your computer to read documents, chats, and web pages and extract interesting information? Wouldn't you like your computer to find connections between documents or your friends profiles based on the meaning contained in the natural language text? What about classifying your writing according to topic, mood, and quality? This tutorial will teach you how to use python to do all this and create your own text butler.

## Audience

Amateur and professional data scientists who want to play with natural language processing and machine learning algorithms in python.

## Objectives

Attendees will build a tool capable of extracting useful information form 1000's of text files per minute on their laptop. They will decode  authorship, topic, style, and language (e.g. English vs Spanish **and** python vs. javascript). They will understand the advantages and disadvantages of several natural language processing tools and how best to combine them into a useful natural language processing pipeline: regexes, bag-of-words, PCA/LSI, K-means, LDA, SVMs, word-vec, and neural nets.

## Detailed Abstract (needs work)

Participants will explore several practical applications of natural language processing to interesting, personally-relevant problems. Participants will first use natural language processing to identify connections between Political Action Committees in Oregon based on their "cause" descriptions. They will then discover anomalies in those connections based on financial transaction data. The resulting pipeline will then be used to classify and index natural language text on the participant's laptops: e-mails, notes, word-documents, e-books, etc.

A brute force word frequency analysis ("bag of words") approach will be pushed to its limit and then dimension reduction will be employed to stretch the scalability of this approach as far as it can go on a non-distributed environment (laptops). Semantic analysis will then be employed. A Word Net graph will be constructed, and augmented with proper nouns from the text. This augmented Word Net will then be used to calculate the similarity of word meanings to augment word frequency statistics. From this distance matrix and graph, participants will compute "factors" describing the sentiment of the documents and other latent characteristics. Participants will target subjects within those documents that interest them and discuss their results. Finally, participants will configure a chat bot plug-in to respond to natural language commands and queries and behave in surprisingly complex ways.

## Outline (needs a lot of work to look feasible)

1. (5 min) what is natural language 
    1. a system of utterances invented by humans "spontaneously" over millions of years.
    2. unstructured text is a generalization of natural language text and the terms are often used interchangeably
    3. natural language is often embedded in structure text (formal languages), like HTML, XML, YAML, SQL, and of course Python as the content of variables, elements, or strings
    4. Examples (why NLP is challenging):
        - HTML tag contents, e.g. the "PyCon 2015..." in <title>PyCon 2015 in Montréal | April 8th – April 16th</title>
        - A textbook, encyclopedia or Wikipedia articles with headings, page numbers, footnotes, etc
        - A social network feed (twitter, facebook, ), e.g. "Brushed my teeth today"
        - A legal contract, license agreement (EULA), annual report, patent "By checking this box you sign away your rights to sue us."
        - Notes to yourself: "Don't forget to take Plato for a walk"
        - Chat room correspondence: "OMG dont be sucha troll!!!"
        - Numbers and prices (e.g. "200 pythonistas", "$50K per year", "1 GB")
    5. Nonexamples
        - HTML and CSS tags
        - python script (but some strings within it may be NL)
        - A CSV file (but some strings within the fields may be NL)
        - mathematical equations (but the integers and fractions within it can be processed as NL)
        - a database (but the names of the fields and tables may be processed as NL)
2. (1 min) what is natural language processing 
    1. Computer processing of languages to do something useful or fun
3. (10 min) why is natural language processing useful and fun 
    2. Example applications
        a. sentiment analysis of customer service data (SAP)
        b. sentiment analysis for trend and finance prediction on twitter and other news feeds (Thomson Reuters)
           - Reuters provides a low-latency feed to hedge funds containing a single bit associated with a stock symbol -- positive or negative impact on price
        c. hardware performance trends based on technician inspection comments (Sharp Electronics Corporation)
        d. enable artificial intelligence agents to train/teach themselves (CMU's NELL)
        e. data migration (ETL) between bodies of structured text like CSV, HTML tables to save the planet (DOE and Building Energy)
4. (5 min) what does artificial intelligence have to do with NLP 
    1. Turing defined it as being able to imitate a human's ability to converse in  natural language text
    2. In some ways coding languages, structured text, and data structures, are just a subset or specialization of natural languages (because they are meant to be written and read by humans *AND* machines)
    3. semantic processing (state of the art NLP) extracts knowledge or meaning from text  
5. (5 min) Context:
    1. what is context
    2. why is it important?
    3. Some common layers of context and meaning
        1. word (the "meaning" of syllables depends on the word they are used in)
        2. compound word ("boot" means something different in "bootstrap" and "boot up")
        3. phrase (noun-phrases are particularly "atomic")
        4. sentence (a sentence can often be presumed to have some grammatically-required elements like a noun and a verb)
        5. paragraph (paragraphs often have an intro, body, conclusion with different word usage assumptions)
        6. passage (quotes, excerpts)
        7. page (text often will refer to images or quotes on the same page, like "see above")
        8. section (topics are changed between sections of an article or book)
        8. chapter (authors change viewpoint/location/subject between chapters)
        9. book (terms and symbols used in a dictionary may only be relevant there)
        10. corpus (a subset of language usages will always have sample biases)
        11. language ("taco" means something different in English than in Spanish)
        12. tribe/city/region ("Zoobombing" means something completely different in Portland than in a war zone)
        12. nation (culture)
        13. planet (yes, projects like SETI are very concerned with NLP of ET languages)
5. (15 min) Getting Started (Setting up a Development Environment:
    1. OSX and Linux instructions for installing python and the packages listed above in the "Environment Preparation" section 
6. (10 min) Coffee break
    1. will continue to help those with trouble getting an environment set up, but will move on with the tutorial session at the conclusion of the break, regardless
7. (10 min) Acquiring a Corpus
    1. using `nltk` to download text corpora (text documents or strings)
    2. extracting text and semi-structure text (tables) from web pages using Scrapy and Beautiful Soup
    for  with some common tools for "quantifying" and structuring unstructured text
8.  (20) Frequency analysis of US President inaugural speeches ()
    1. segmentation/tokenization/parsing
        - characters (encoding issues, some natural languages like Japanese Kanji and Chinese don't have "letters")
        - words
            - digits and symbols and unicode as part of words
            - punctuation at the end of sentences and word
            - hyphenation
            - typos
            - spelling variations (British English)
            - language variations (Spanish, French, slang)
        - bag-of-words counting (frequency analysis) ignores context at any layer above the "documents"
        - agnostic counting
    2. stemming
        - nltk stemmers
    3. counting 
        - Data structures like `collections.Counter` that discard context/order 
        - Can `collections.OrderedDict` be used to preserve context and order? (not easily)
    4. normalization of counts/frequencies/probabilities
    5. occurrence matrices ("word space" or "word vector space" in information theory)
        - uses for word-word, word-document, document-word, and document-document matrices
        - "word space" is a way of giving words a distance metric, from each other as individuals and as collections of words (documents)
            - Leventshtein distance
                - Distance
            - statistical (frequency) word space
                - nltk.metrics.distance.jaccard_distance
                - nltk.metrics.distance.masi_distance
                - nltk.metrics.distance.presence
            - direct semantic word space (we'll talk about WordNet later)
            - syntactic/gramatical word space (we'll talk about POS tagging later)
            - statistical nltk distance measures/metrics:
    2. complexity/entropy/information measures for unstructured text
        a. compression ratio
        b. entropy
        c. predictability (human trials by Claude Shannon et al.)
9. Dimension reduction (PCA or SVG)
    1. occurrence matrices will grow to become impractical
        - 100k words/tokens counted across 10k documents = 1 GB of data, if stored efficiently
        - ignoring "stop words" and low-information-content words won't significantly reduce the dimensions
        - many machine learning algorithms are impractical at this scale:
            - decision trees
            - KNN
            - K-means
            - Support vector machines
        - SVD (PCA) can reduce the dimensions and enable many powerful machine learning algorithms to be employed
        - When SVD is impractical (e.g. 100k x 100k matrices or larger), dimension reduction can be based on the entropy found in each word and document independent of the others
    2. ntlk US inaugural presidential speech word-frequency example
        - raw occurrence matrices
        - reduced-dimension occurrence matrices
    3. d3 visualizations of occurrence matrices
        - as "checkerboard" grids or heat-maps
        - as graphs or networks (D3 force-directed graph)
9. (10 min) Getting Fuzzy
    1. regular expressions
        - examples for use in a chat bot
        - examples for use in a crawler for financial information
        - what they're good at (semi-structured text) and what their not good for (not robust/reliable)
    2. fuzzywuzzy (uses "quick" Levenshtein distance)
        - examples for matching database table/column names
        - when you need the "best" match and you need it fast
    3. fuzzy regular expressions (`regex` package)
        - example use in a chatbot (`will`)
        - when you want the very "best match" and you can wait
10. (10 min) Knowledge extraction
    1. date/time information using python-dateutil
        - `will` example "remind me to knock off at 5"
    2. regexes to extract prices
11. (10 min) sentiment analysis to gage chat room "mood"
    1. `will` chatbot example using nltk
12. (10 min)sentence structure
    1. nltk POS tagging tools and examples
12. (10 min) Semantic processing
    1. nltk WordNet interface
    2. use NLTK to populate a simple knowledge base about you based on your hard drive contents


[Draft Slides that Will reuse much of the Markdown above](http://hobson.github.io/pycon2015-nlp-tutorial/docs/slidedeck-tutorial/index.html#1)


Example Material, much of which will be updated and incorporated into this tutorial

[Material previously-presented at a PDX-Python user-group meeting](http://hobson.github.io/pug/pug/docs/slidedeck-pdxpy/index.html#1)

Example Visualizations after dimension reduction to only the 100 Highest Entropy Words

The co-occurrence matrices for US Presidential Inaugural Speeches can be visulized as heat-maps and shuffled/sorted according to various criteria, like political party of the president, or year of speech:
[Word Co-Occurrence Matrix Visualization and Sorting](http://hobson.github.io/pug/pug/miner/static/word_cooccurrence.html)
[Document Similarity Matrix Visualization and Sorting](http://hobson.github.io/pug/pug/miner/static/doc_cooccurrence.html)

Can you guess what will happen if you produce a force-directed graph that includes both words and documents? The strength of connections between nodes (their attraction, or similarity, or inverse distance metric) is their cooccurrence frequency.
[Graph Clustering of Words and Documents](http://hobson.github.io/pug/pug/miner/static/occurrence_force_graph.html)

Can you guess the words that will be outliers (usage is independent of other words) in innaugural speeches?
[Word Co-Occurrence Graph Clustering](http://hobson.github.io/pug/pug/miner/static/word_force_graph.html)

Can you guess the presidential innaugural speeches that will be outliers when they are clustered according to word usage?
[Document Similarity Graph Clustering](http://hobson.github.io/pug/pug/miner/static/doc_force_graph.htm)

[Material previously-presented at a PDX-Python user-group meeting](http://hobson.github.io/pug/pug/docs/slidedeck-pdxpy/index.html#1)

Example Visualizations of US Presidential Inaugural Speeches and their 100 Highest Entropy Words

The co-occurrence matrices can be visulized as heat-maps and shuffled/sorted according to various criteria, like political party of the president for US innaugural speeces:
[Word Co-Occurrence Matrix Visualization and Sorting](http://hobson.github.io/pug/pug/miner/static/word_cooccurrence.html)
[Document Similarity Matrix Visualization and Sorting](http://hobson.github.io/pug/pug/miner/static/doc_cooccurrence.html)

Can you guess what will happen if you produce a force-directed graph that includes both words and documents? The strength of connections between nodes (their attraction) is their cooccurrence.
[Graph Clustering of Words and Documents](http://hobson.github.io/pug/pug/miner/static/occurrence_force_graph.html)

Can you guess the words that will be outliers (usage is independent of other words) in innaugural speeches?
[Word Co-Occurrence Graph Clustering](http://hobson.github.io/pug/pug/miner/static/word_force_graph.html)


Can you guess the presidential innaugural speeches that will be outliers when they are clustered according to word usage?
[Document Similarity Graph Clustering](http://hobson.github.io/pug/pug/miner/static/doc_force_graph.htm)

