import sys

file = open(sys.argv[1], 'r')
file2 = open('Q1.txt', 'w')

wordcount = {}
wordlist = []
index = 0

for word in file.read().split():
    if word not in wordcount:
        wordcount[word] = 1
        wordlist.append(word)
    else:
        wordcount[word] += 1

for i in wordlist:
    # print (i, index, wordcount[i])
    if index < len(wordlist) - 1:
        file2.write("%s %s %s\n" % (i, index, wordcount[i]))
        index += 1
    else:
        file2.write("%s %s %s" % (i, index, wordcount[i]))

file.close()
file2.close()