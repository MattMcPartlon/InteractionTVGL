root = "/home/mmcpartlon/suppdata/converted_aln"
out_root='/home/mmcpartlon/suppdata/meff_aln'
acceptable_chars ='ACDEFGHIKLMNPQRSTVWYXBZO-'
seq_rt = '/home/mmcpartlon/suppdata/seq'
acceptable_chars_set = {x for x in acceptable_chars}
import os
if not os.path.exists(out_root):
    os.makedirs(out_root,exist_ok=True)
for x in os.listdir(root):
    pth = os.path.join(root,x)
    del_out_pth = os.path.join(out_root,x)
    out_pth = os.path.join(out_root, x.split('.')[0]+'.a2m')
    if os.path.exists(out_pth):
        os.remove(out_pth)
    L = -1
    seq = x.split('.')[0]
    seq_path = os.path.join(seq_rt,seq+'.fasta')
    with open(seq_path,'r+') as f:
        for d in f:
            if d.startswith('>'):
                continue
            else:
                seq = d
    with open(pth,'r+') as fin:
        comment_line = ''
        seq_line = ''
        with open(out_pth,'w+') as fout:
            fout.write('>target\n')
            fout.write(seq if seq.endswith('\n') else seq+'\n')
            parsed_seq=False
            for line in fin:
                data = ''
                if line.startswith('>'):
                    parsed_seq = False
                    data = line.strip().split(' ')[0]
                    data = data.split('-')
                    comment_line = data[0]+data[1]+'\n'
                else:
                    parsed_seq = True
                    tmp = line.strip()
                    tmp_hash = {a for a in tmp}
                    if tmp_hash.intersection(acceptable_chars_set)!=tmp_hash:
                        if 'U' not in tmp_hash.difference(acceptable_chars_set):
                            assert False
                    l = len(tmp)
                    if L<0:
                        L = l
                    assert l == L
                    seq_line = line
                    if 'U' in tmp_hash:
                        seq_line = line.replace('U','X')
                if parsed_seq:
                    if seq_line!=seq:
                        fout.write(comment_line)
                        fout.write(seq_line)
    print('finished ',fout)


