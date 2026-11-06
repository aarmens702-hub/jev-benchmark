---
layout: default
title: "I bet on Jev in public. Here is what a pre-registered test showed."
---

# I bet on Jev in public. Here is what a pre-registered test showed.

*Aarmen Sidhu, 6 November 2026*

Recently I left a comment on a LinkedIn post about Jev, a decision model that takes a list of options with each request and returns a choice with a confidence score. I argued it was more than a "rebranded classifier". A senior ML engineer replied with a fair challenge: zero-shot classifiers, embedding classifiers and constrained language models already handle options given at request time. The real test, they said, is Jev against optimized lightweight classifiers, measuring cost, latency, calibration and accuracy at equal automation coverage.

So I ran that test. To keep myself honest, I wrote down the whole plan first, including the code, the settings and a table saying what each possible result would mean. I published it on GitHub before loading any test data. The result would be read off that table, whichever side it favoured.

## The setup

The main test is Banking77: 3,080 real customer banking messages, each belonging to one of 77 intents. I compared Jev 1.14, given no labelled examples, against a simple classifier (logistic regression on sentence embeddings) trained with 1, 2, 3, 5, 8, 16, 35 or all of about 130 examples per intent.

I scored each method on the 80% of messages it was most confident about. That is how a routing system works in practice: it acts on confident cases and sends the rest to a person or a bigger model. I also tested the three older approaches the engineer named, and an easier dataset, AG News (4 news topics), as a side check.

## What it showed

![Gap between the classifier and Jev](figure1.png)

The pre-written verdict: **Jev leads through 3 labels per intent, and the classifier leads from 8. Jev's value on this task is limited to cold starts.**

- With no examples, Jev scored 90.6%.
- The classifier scored 86.5% with 3 examples per intent, 91.4% with 5 (no clear difference) and 94.3% with 8.
- 8 examples per intent is 616 labelled messages in total. Past that, the classifier only pulls further ahead, to 98.7% with everything.

So the engineer was right on the main point. On this task, a few hundred labels and a free classifier beat Jev.

## Where Jev held up

- **The older zero-shot approaches were not close.** Embedding search scored 78.5%, a small language model forced to pick from the list scored 62.0%, and a zero-shot NLI classifier scored 61.6%. All three were 12 to 29 points behind Jev. They do accept options at request time, but on this task they were not a substitute.
- **Jev uses examples well.** Given the same 8 examples per intent, Jev scored 95.6%, at least as good as the classifier trained on them.
- **It followed a changed rule.** In a small made-up routing test, I swapped the routing rule halfway. Jev, given the new rule as text, scored 97.1%, close to a classifier plus the rule written as code (98.4%). The small language model and NLI classifier scored 75.5% and 79.6%.
- **On easy topics it lasted longer.** On AG News, Jev stayed ahead until 16 examples per topic. So the Banking77 result does not carry over to every task.

## Cost, speed and confidence

Jev cost about US$0.10 per 1,000 messages and answered in about 0.18 seconds. The small language model took about 1.6 seconds. Jev's confidence matched its accuracy better than the classifier's did at 8 examples, but it said exactly 1.0 on 42% of messages, which makes it hard to rank its answers finely. The whole test cost US$1.59.

## What I got wrong

My comment said a traditional classifier "is trained on fixed labels for one task". The engineer had named methods that are not fixed-label, so that point did not stand. I also said "most agent stacks" send every routing decision to a language model. I have no evidence for that, so I withdraw it.

## Limits

- One model version, Jev 1.14.
- One main dataset.
- I had seen earlier informal results before writing the plan, so the design is pre-registered but not blind.
- Jev's training data is not public, so it may have seen these public datasets.

Everything is in the [repo](https://github.com/aarmens702-hub/jev-benchmark), with the plan frozen at tag [`freeze-v1`](https://github.com/aarmens702-hub/jev-benchmark/tree/freeze-v1) before any test data was loaded: the frozen plan, every prediction and one command to rebuild every number.

*I used Claude, an AI assistant, to help design, code and check this study and to draft this post. I reviewed the design and every number.*
