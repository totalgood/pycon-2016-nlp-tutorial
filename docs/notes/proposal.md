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

### Introduction (10 min)

- Interesting NLP applications
    - chatbots
    - behavior modification
    - natural language generation
- state of the art NLP capabilities
    - skip-grams and Word-Vector math
        - teaser: "king" - "man" + "woman" = "queen"
    - Google Now & Siri
        - description of pipeline (we will build some elements here)

### Feature Extraction with Python (30 min)

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

### Workshop: Feature Extraction Pipeline (20 min)

Students will use the tools provided in the presentation to build a python function capable of processing 10's of thousands of tweets in a few minutes to produce meaningful clusters based on tweet content.

### Feature Simplification (40 min)

- Feature Reduction
    - Calculating entropy (information value) with `numpy`
    - Principle Component Analysis with `scikit-learn.PCA`
        - how it works (overview of the matrix algebra behind the scenes)
        - where it works best
        - what to watch out for
- Plotting and Exploring
    - scipy scatter matrix plots
        - visulizing natural language feature vectors
        - projecting/slicing
    - `json.dumps` of TFIDF matrices for d3.js matrix visualizations
    - using python to manipulate nested dicts to create json required for interactive d3.js force-directed graphs 

### Workshop (20 min)

Attendees will use the tools provided to process and plot twitter tweets to find more informative clusters and patterns.

### Supervised Learning (40 min)

- Extracting statistics about tweets
    - timing
    - following the trail of retweets
    - favorites
    - identifying influential "likers"
- Plotting and Exploring
    - time series plots with `pandas.DataFrame.plot()`
    - adding "category" to scipy scatter matrix plots
    - plots to show why PCA falls short
- Improving on PCA with supervision (labeled tweets)
    - Linear Discriminant Analysis with `scikit-learn.LDA`
    - clustering using K-means in `scikit-learn` 

### Workshop (20 min)

Attendees will use the tools provided to process and plot twitter tweets and augment their pipeline to improve their prediction accuracy.

### Discussion (30 min)

As time permits we will answer questions and discuss how these techniques are being used for a state-of-the art natural language pipeline employed at Talentpair. We employ "Hierarchical Context-Partitioned Word Vectors" to match candidates seeking work (in the form of resumes) and opportunities offered by employers (in the form of role descriptions).

### Additional Notes

Both speakers have presented at PyCon in the past and Hobson has spoken about natural language processing on 5 previous occasions and have a track record of teaching novices to use natural language processing techniques in a short amount of time. Hobson served for years as a mentor for Georgia Institute of Technology grad students in Machine Learning and am currently mentoring [SlideRule](mysliderule.com) students. My presentations are very interactive and I try to engage participants individually throughout a tutorial or presentation by soliciting their ideas and provoking their critical thinking.

