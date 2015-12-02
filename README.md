# Making Connections with Natural Language Processing

This repository holds Work-In-Progress material intended for a PyCon 2016 NLP Tutorial based on [this proposal](docs/notes/proposal.md).
### Description

Do you dream of teaching your computer to read documents, chats, and web pages to extract information? Wouldn't it be cool if you could use Python to find connections between documents on your laptop or between your people's profiles and tweets on twitter? What about classifying your own writing according to topic, mood, and quality? This tutorial will teach you how to use python to do all this, creating your own personal text butler.

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

### Instructor Notes

2. [Language](docs/notes/language.md)
    - [formal language](jupyter/formal-language.ipynb)
    - [text adventure](scripts/adventure-fsm.py)
    - [relaxing the rules](jupyter/relaxing-rules.ipynb)[Chomsky-img]
    - [natural language](jupyter/natural-examples.ipynb)
    - [using regular expressions](jupyter/eliza-like.ipynb)
3. [Ambiguity](docs/notes/ambiguity.md)
    - [stemmers](jupyter/stemmers.ipynb)
        - morphological parsing goal
        - porter stemmer problems
    - [transducers](jupyter/transducers.ipynb)
        - [sequential transducers](jupyter/sequential-transducers.ipynb)
        - [subsequential transducers](jupyter/subsequential-transducers.ipynb)
    - Brute Force (Enumeration)
        - Inefficient or English and Japanese
            - [millions of words and meanings][Michel]
        - impractical for Turkish and other complex morphotactics
            - [trillions of possibilities?][Jurafsky09-46]
10. [References](docs/notes/references.md)

[Jurafsky09-46]: http://stp.lingfil.uu.se/~santinim/ml/2014/JurafskyMartinSpeechAndLanguageProcessing2ed_draft%202007.pdf#page=48 "Speech and Language Processing 2nd Edition, DRAFT 2007"
[Chomsky-svg]: https://commons.wikimedia.org/wiki/File:Chomsky-hierarchy.svg#/media/File:Chomsky-hierarchy.svg "Chomsky-hierarchy by creative commons User:J._Finkelstein Licensed under CCA-SA-3.0"
[Michel]: http://www.librarian.net/wp-content/uploads/science-googlelabs.pdf "Quantitative Analysis of Culture Using Millions of Digitized Books by Jean-Baptiste Michel, Erez Lieberman Aiden, et al."

