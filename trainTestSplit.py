f = open('devAndTrainPOS.pos', 'r')
count = 0

for line in f:
    if line == '\n':
        count += 1

f = open('devAndTrainPOS.pos', 'r')
f2 = open('training.pos','w')
f3 = open('testing.pos','w')
count2 = 0

for line in f:
    if(count2 <= int(count*.80)):
        f2.write(line)
    else:
        f3.write(line)

    if line == '\n':
            count2 += 1
f.close()
f2.close()
f3.close()
        
        
