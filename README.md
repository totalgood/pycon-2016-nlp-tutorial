# Making Connections with Natural Language Processing

## Description

Do you dream of teaching your computer to read documents, chats, and web pages to extract information? Wouldn't it be cool if you could use Python to find connections between documents on your laptop or between your people's profiles and tweets on twitter? What about classifying your own writing according to topic, mood, and quality? This tutorial will teach you how to use python to do all this, creating your own personal text butler.

## Contents

1. [Introduction](docs/notes/introduction.md)
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

