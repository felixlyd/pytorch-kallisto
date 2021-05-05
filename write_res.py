from tqdm import tqdm


def write_res(counts, out_file, abundance_file):
    wf = open(out_file, 'w')
    with open(abundance_file, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            line_str = line.strip()
            if not line_str:
                break
            if "target_id	length	eff_length	est_counts	tpm" in line:
                wf.write("target_id	length	eff_length	est_counts	my_counts\n" )
            else:
                gene_name, length, eff_length, est_counts, tpm = line_str.split("\t")
                wf.write("{}\t{}\t{}\t{}\t{:3f}\n".format(gene_name, length, eff_length, est_counts, counts[i-1]))
    wf.close()
