# encoding: utf-8

with open('degree.txt.bac', 'r', encoding='utf-8') as fr:
	lines = fr.readlines()
with open('degree.txt', 'w', encoding='utf-8') as fw:
	for line in lines[3:72]:
		print('%s\t%.1f' % (line.strip(), 2), file=fw)
	for line in lines[74:116]:
		print('%s\t%.1f' % (line.strip(), 1.5), file=fw)
	for line in lines[118:155]:
		print('%s\t%.1f' % (line.strip(), 1), file=fw)
	for line in lines[157:186]:
		print('%s\t%.1f' % (line.strip(), 0.5), file=fw)
	for line in lines[188:200]:
		print('%s\t%.1f' % (line.strip(), -0.5), file=fw)
	for line in lines[202:232]:
		print('%s\t%.1f' % (line.strip(), -1), file=fw)
