# Making Connections with Natural Language Processing

## Description

Do you dream of teaching your computer to read documents and extract information? Would you like to use Python to find connections between text documents on your laptop or tweets on twitter? What about classifying your own writing according to topic, mood, and quality? This tutorial will show you how to build a personal text butler to uncover patterns and trends in natural language text.

## Audience

Amateur and professional data scientists who want to play with natural language processing and machine learning algorithms in python.

## Objectives

Attendees will build a tool capable of extracting  information from text files on their laptop. The tool will be able to classify by authorship, topic and style. In the process of building and training this tool, attendees will gain a practical understanding of the state of the art for NLP, including the technologies behind Siri and Google Now: regexes, bag-of-words, scale-space processing, PCA/LSI, K-means, LDA, SVMs, word2vec, and neural nets.

## Detailed Abstract

### Prerequisites

Students that have built a few python scripts or modules from scratch and have done some file or string processing (search and replace, word counts, regular expressions) will be able to keep up with the pace of the examples in this tutorial. Students familiar with basic statistics concepts like probability distributions, histograms, and perhaps word frequency will have not difficulty following the explanations of the math behind the examples used in this tutorial.

Students will need ipython, NLTK, scipy, scikit-learn, and Pandas installed on their laptop to run all the examples (mostly ipython notebooks) in this tutorial. Students can use pip to install the [requirements here](/requirements.txt). If the user does not have 100 or more text files they would like to analyze then they should use NLTK to download a corpus of documents of interest to them, such as the US President Inaugural Speeches. iPython notebooks that use this NLTK corpus or a folder containing text files on a laptop will both be provided. 

### Overview

The tutorial will be divided into 5 sections.

  0. Introduction and Motivation
  1. Classical Language Processing
  2. Statistical Language Processing
  3. An Application: Unsupervised Document Classification
  4. An Application: Supervised Document Similarity Scoring

#### 0. Introduction and Motivation

Several recent breakthroughs and applications of Natural Language Processing will be described. The technologies required for each will be discussed briefly to motivate the technologies discussed at length in subsequent sections.

#### 1. Classical Language Processing

The first section will explain the types of languages that computers can process, natural and formal (and the Chomsky hierarchy). The mathematical models (regular expressions, finite state machines, state transition tables) used to process languages with a computer will be demonstrated using [iPython notebook examples](https://github.com/totalgood/pycon-2016-nlp-tutorial/blob/master/jupyter/classical-nlp/classical-nlp.ipynb). A simple Finite State Machine implementing a 'text adventure' video game (designed and largely implemented by a 9-yr old girl learning python) will be used as a tool to compare and contrast these languages and the models used to process them. This section will conclude with an explanation of the role these tools play in a modern natural language processing pipeline such as Google Now or Siri.

#### 2. Statistical Language Processing

A brute force word frequency analysis ("bag of words") approach will be pushed to its limits. Dimension reduction techniques (SVD, PCA, LSI. LDA) will be used to extend and generalize the performance of this statistical approach. Finally a hierarchical or scale-space approach to document processing will be demonstrated and students will learn how to extend their statistical models produce a more subtle and accurate characterization or scoring of documents.

A Word Net graph will be constructed, and augmented with proper nouns from the text. This augmented Word Net will then be used to calculate the similarity of word meanings to augment word frequency statistics. From this distance matrix and graph, participants will compute "factors" describing the sentiment of the documents and other latent characteristics. Participants will target subjects within those documents that interest them and discuss their results. Finally, participants will configure a chat bot plug-in to respond to natural language commands and queries and behave in surprisingly complex ways.

#### 3. Application: Unsupervised Document Clustering

The natural language processing pipeline created in the previous sections will be used to preprocess and index a set of text documents on the participants' laptops (these can be downloaded using NLTK if no documents of interest already exist on a participant's laptop). Those that do not have documents they want to process on their laptops can follow along with the examples using the US Inaugural Speeches provided by NLTK and used for [the instructors examples](http://hobsonlane.com/pug/pug/docs/slidedeck-pdxpy/index.html#1). These preprocessed documents will be classified using SVD and K-means to identify themes and subject matter.

The fundamental limitations of unsupervised dimension reduction and clustering will be explained using plots of the results from this example.

#### 4. Application: Supervised Document Scoring and Pairing

Participants will explore unstructured text descriptions of political action committees and use financial transaction data from these committees as labels for a supervised learning algorithm. LDA and Synthetic Annealing (Force-Directed Graphs) will be used to classify these text descriptions and [show connections](https://github.com/totalgood/hackor/tree/master/jupyter) between seemingly unrelated committees. Anomalies and trends will be shown in matplotlib and interactive D3 plots. These same algorithms will be applied to the participant-relevant text files utilized in the previous section (files on the participants laptops). Participants will encouraged to share with the group any interesting results they find for the documents they chose to model and mine.

### Outline

1. [Introduction and Motivation](docs/notes/introduction.md)
    1. Applications
        1. [bioinformatics](https://en.wikipedia.org/wiki/FASTA_format) and [DNA analysis](http://pythonforbiologists.com/index.php/introduction-to-python-for-biologists/regular-expressions/)
        2. conversational assistants (Siri, Google Now)
        3. Telephone Airline Reservation Systems
        4. Chat Bots
        5. MMORPG Chat Chaperoning
        6. Web Search
        7. Document Search/Indexing
        8. Sports Reporting
        9. Financial Market Reporting
        10. [Forensic Linguistics (Authorship Attribution)](http://www.slideshare.net/PyData/authorship-attribution-forensic-linguistics-with-python-scikit-learn-pandas-kostas-perifanos)
        10. Literature Analysis (Shakespeare Authorship, Historical Social Norms)
        11. [Plagerism Detection](https://www.academia.edu/9984589/PLAGIARISM_DETECTION_ALGORITHM_USING_NATURAL_LANGUAGE_PROCESSING_BASED_ON_GRAMMAR_ANALYZING)
        12. [Python Pattern and Anti-pattern Classification](https://www.quantifiedcode.com/knowledge-base/#python)
        13. NOAA Weather Reports
        14. Road Condition Reporting
        15. [Online Customer Service](http://dl.acm.org/citation.cfm?id=1643823.1643908)
        16. Spelling and Grammar Correction
        17. Smartphone Keyboard Entry Prediction
        18. [Machine Translation](http://www.hutchinsweb.me.uk/Nutshell-2005.pdf)
    2. Technologies
         1. Regular Expressions (Segmentation)
         2. Transcoders (Stemming)
         3. PCA & LDA (Dimension Reduciton)
         4. Neural Nets (NL Generation, Dialog Engines/Chatbots)
2. Languages
    1. Mathematical Language (not important, just want you to be aware of it)
        1. A set (can be infinite) of symbol sequences
           - Example: The set of all possible Python programs
    2. "Real" Language:
       - The rules that govern the processing of a sequence of symbol sequences
          - symbols, spelling
          - grammar, syntax, punctuation
          - conventions, slang, ambiguity
       - This is a nongenerative understanding of the concept "language"
          - We think computer's understand the language "Python"
          - Computers can't write a meaningful python program...yet.
    3. Programming Languages are a Subset of Natural Languages
        1. Symbol sequences are the result of sequence of decisions
        2. The rules that constrain those decisions are the grammar
        3. A brain (intelligence) leaves its fingerprint in those decisions 
    4. A text-adventure as a [Regular Language](https://github.com/totalgood/pycon-2016-nlp-tutorial/blob/master/jupyter/classical-nlp/classical-nlp.ipynb)
        1. Regular Expressions
        2. Finite State Machines
        3. State Transition Matrices
        4. Language Generation with a Regular Expression
        5. Regular Expression Grammar
    5. Chomsky hierarchy (simplest first)
        1. Regular Languages (choose-your-own-adventure books/games)
        2. Context-free Languages
        3. Context-sensitive Languages (text adventure games)
        4. Recursively Enumerable Languages (Python, C++, if it didn't allow infinite loops)
    6. Uses for Regular Expressions and Transcoders in Language Processing
        1. Word & [sentence segmentation](https://github.com/hobson/nlup)
        2. Stemming
        3. Part of Speech Tagging
        4. Morphotactics
    7. [Autocode a regular expression in python](jupyter/autocode-regex.ipynb)
3. Statistical Language Processing
    1. Word Frequency Analysis
    2. High-Dimensional Vector Distance Metrics
        1. p-norm
        2. cosine similarity
        3. cartesian distance
        4. manhattan distance
        5. supremum distance
    2. Dimension reduction
        1. SVD/PCA/LSI Suboptimality
        2. LDA Requires Labeled Data
    3. Scale-Space Frequency analysis
    4. WordNet
    5. Word2Vec
4. Application: Unsupervised Document Clustering
    0. K-means
    1. [Force-Directed Graphs](http://hobsonlane.com/pug/)
    2. [Cooccurence Matrices](http://hobsonlane.com/pug/)
    3. [Python Used to Construct Force-Graphs](http://hobsonlane.com/pug/pug/docs/slidedeck-pdxpy/index.html#1)
5. [Application: Supervised Document Scoring and Pairing](http://totalgood.github.io/talks/2015-10-27-Hacking-Oregon-Hidden-Political-Connections.html#/)

### More Info

This tutorial will conclude with an explanation of a state-of-the art NLP technique used a Talentpair. Most of the participants will then emerge with an understanding of at least one state-of-the-art approach for machine learning using natural language data. Our approach is called "Hierarchical Context-Partitioned Word Vectors," which is just a fancy way of saying that you create n-grams that start with a context key prefix. You "tag" your tokens with their context. This creates a nested mapping (nested python dict) of tokens which can be flattened into n-grams. So a the 1-gram token "fun" from a facebook profile, might be tagged with its context to create an 3-gram "social:fb:fun" while a Wikipedia article with the same word might become "reference:wikipedia:fun".  Of course this approach has the disadvantage of expanding your dimensions (vocabulary of words). It also prevents you from directly making unsupervised matches across contexts (e.g. Facebook profiles paired up with famous people in Wikipedia articles). However, once you perform PCA you can often find unsupervised matches or clusters across domains. And this approach has the advantage of making supervised match learning possible with LDA to create combinations of dimensions across the contexts that  reduce the dimensions and generalize your model across domains.  PCA and LDA would be required even without this context-tagging.

### Additional Notes

I've spoken about natural language processing using python on 5 previous occasions and have a track record of teaching novices to use natural language processing techniques in a short amount of time. I've served for years as a mentor for Georgia Institute of Technology grad students in Machine Learning and am currently mentoring [SlideRule](mysliderule.com) students. My presentations are very interactive and I try to engage participants individually throughout a tutorial or presentation by soliciting their ideas and provoking their critical thinking.

