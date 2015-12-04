# Making an Impact with Python Natural Language Processing Tools

## Description

Do your tweets get lost in the shuffle? Would you like to predict a tweet's impact before you hit send? Python now has all the tools to make this possible. Several Python packages for machine learning and natural language processing have reached "critical mass" and can now be combined to perform these and other powerful natural language processing tasks. This tutorial will teach you how.

## Audience

Amateur and professional data scientists who want to learn about a powerful combination of python tools and techniques for natural language processing

## Objectives

Attendees will build a python module that can determine the best time of day to tweet on a particular subject. While building this tool, attendees will become familiar with the most powerful combination of python packages for performing state-of-the-art natural language processing.

## Detailed Abstract

### Prerequisites

Students that have written python scripts, modules, or a package and are familiar with the basic string manipulation and formatting capabilities built into python will have the necessary skill to complete this tutorial. In addition, any students who are familiar with linear algebra, and basic statistics concepts (like probability and variance) will be able to grasp the mathematics behind the tools assembled during the tutorial, but this is not required.

### Python Development Environment

Students will need iPython, NLTK, scipy, scikit-learn, and Pandas installed on their laptops to run the examples in this tutorial and build the tweet impact predictor tool. Students can use pip to install the [requirements listed here](/requirements.txt). In addition students will need to install the python twitter api or download a 50 MB compressed file of tweets in order to train and test their tweet predictor.

### Overview

Participants will develop a tweet natural language processing pipeline in three modules. The first section of the pipeline will be a natural language feature extractor and normalizer based on python builtins `collections`, `string`, and `re` combined with the powerful Pandas `DataFrame` data structure. The second section will utilize `scikit-learn` and `numpy` to simplify the feature set to a manageable number of features. It will find optimal combinations of reduced numbers of features that provide the greatest information about the subject matter of the tweets being processed. The final section of the pipeline will assemble a training set based on tweet statistics not contained in the natural language content of the tweets and combining this with the natural language features to cluster and classify tweets according to their popularity (number of favorites), and reach (number of potential viewers due to retweets). A neural net will be trained to predict tweet impact (popularity and reach) based on the time of day and day of week as well as the tweet text planned to be sent.


## Outline

### Introduction (10 min presentation)

- Interesting NLP applications
    - chatbots
    - behavior modification
    - natural language generation
- state of the art NLP capabilities
    - Google Now & Siri

#### Language Manipulation with Python (30 min presentation)

- `str.split` to quickly extract words from a tweet
- `collections.Counter` to count word occurrences
- Explore regular expressions in a text adventure
    - Text Adventure games vs. Choose Your Own Adventure books
    - Python Regular Expressions vs. Memoryless Regular Expressions 
    - `re.split` to more accurately extract words (tokens)
    - `nltk` stemmers
    - `nltk` part-of-speech tagging
    - `nltk` word root parsers
    - `nltk` stop word filters
- `pandas.Series` and `pandas.DataFrame`
- `np.linalg.norm` and `np.dot` to efficiently normalize word counts and frequencies
- `scipy.TFIDF` to efficiently store normalized word frequencies in a sparse matrix
- `np.linalg.norm`, `np.dot` to compute "distances" between tweets
- `scikit-learn` Clustering/grouping like tweets together 

#### Feature Extraction Pipeline (20 min workshop)

Students will use the tools provided in the presentation to build a python function capable of processing 10's of thousands of tweets in a few minutes to produce meaningful clusters based on tweet content.

### Feature Simplification

- Feature Reduction
    - `scikit-learn.PCA` 


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

