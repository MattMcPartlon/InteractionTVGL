import os
import re
import subprocess


def parse_seq_name(line):
    line = line.replace('>', '')
    try:
        return re.match(r'seq-\d+', line)[0]
    except Exception as e:
        print('error parsing for sequence name in line :', line)
        raise e


def parse_seq_names_in_cluster_file(cluster_file_path: os.PathLike):
    clusters = []
    curr_cluster = []
    with open(cluster_file_path, 'r') as f:
        for line in f:
            if line.startswith('>Cluster'):
                if curr_cluster:
                    clusters.append(curr_cluster)
                curr_cluster = []
            else:
                # o.w. we are inside a cluster, and should parse for the sequence name
                curr_cluster.append(parse_seq_name(line.split('>')[1]))
    return clusters


def get_name_to_seq_map(input_aln_path: os.PathLike):
    # add sequences to a map with key = name
    key = None
    seq_map = {}
    in_comment = False
    with open(input_aln_path, 'r+') as f:
        for line in f:
            if line.startswith('>'):
                # line contains sequence name and description
                key = parse_seq_name(line) if not in_comment else key
                in_comment = True
            else:
                # line corresponds to sequence
                in_comment = False
                seq_map[key] = line.strip()
    return seq_map


def gen_cdhit_clusters(cd_hit_seqs_path: str,
                 output_cluster_path: str = './tmp.clstr',
                 cd_hit_options = '-c 0.8, -G 1 -n 5 -g 0 -aS 0.85 -aL 0'):

    """
    :param cd_hit_seqs_path:
    :param msa_alignment_path:
    :param output_cluster_path:
    :param use_global_identity:
    :param seq_id_thresh:
    :param word_len:
    :param fast_clust: 0 for fast (less accurate default), 1 o.w.
    :return:
    """

    if not cd_hit_seqs_path.endswith('.fasta'):
        raise Exception(f'alignment path must be in fasta format! Got : {cd_hit_seqs_path}')
    cmd = f'cd-hit -i {cd_hit_seqs_path} -o {output_cluster_path} {cd_hit_options}'
    subprocess.call(cmd, shell=True)
    # parse the cluster file and grab the sequences/cluster
    output_cluster_path += '.clstr'
    names_per_cluster = parse_seq_names_in_cluster_file(output_cluster_path)
    msa_alignment_path = cd_hit_seqs_path
    name_to_seq_map = get_name_to_seq_map(msa_alignment_path)
    n_clusters = len(names_per_cluster)
    return {cid: [name_to_seq_map[name] for name in names_per_cluster[cid]] for cid in range(n_clusters)}
