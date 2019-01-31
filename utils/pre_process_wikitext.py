# -*- coding: utf-8 -*-
""" Script to pre-process the wikitext2 dataset. """
import codecs
import nltk


def pre_process(filename, check=True):
    """ Split lines into wikitext2 to their sentences. """
    nltk.download('punkt')
    ntokens = 0
    if check:
        lines = []
        for line in codecs.open(filename, "r", "utf-8"):
            # lines += line.replace("\n","").split() + ["<eos>"]
            lines += line.split() + ["<eos>"]
        ntokens = len(lines)

    lines = []
    for line in codecs.open(filename, "r", "utf-8"):
        lines.append(line.replace("@.@", "&numb") + "<eos>")

    split_lines = [nltk.sent_tokenize(line.strip()) for line in lines]

    new = []
    for line in split_lines:
        if len(line) > 1:
            new_lines = line[:-2]
            new_line = line[-2] + " " + line[-1]
            new_lines += [new_line]
        else:
            new_lines = line
        new += new_lines

    split_lines = [line.replace("&numb", "@.@")
                   for line in new]

    if check:
        len_new = [len(l.split()) for l in split_lines]
        assert ntokens == sum(len_new)
        print("# of tokens in the original file = {0}\n"
              "# of tokens after pre-processing = {1}\n".format(
                  ntokens, sum(len_new)))

    output = "{0}.sents".format(filename)
    print("Saving pre-processed file to {0}\n".format(output))
    with codecs.open(output, "w", "utf-8") as f:
        for line in split_lines:
            f.write("{0}\n".format(line.replace("\n", "")))


if __name__ == "__main__":

    train = ".data/wikitext-2/wikitext-2/wiki.train.tokens"
    valid = ".data/wikitext-2/wikitext-2/wiki.valid.tokens"
    test = ".data/wikitext-2/wikitext-2/wiki.test.tokens"

    print("Pre-processing training set...")
    pre_process(train)

    print("Pre-processing validation set...")
    pre_process(valid)

    print("Pre-processing test set...")
    pre_process(test)
