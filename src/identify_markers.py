#
# Python 3.x
#
import sys, gzip
from library_utils import compare_background_vs_cancer_alpha_value_distribution_files, compare_background_vs_cancer_alpha_value_distribution_files_with_memory_saving_way, write_alpha_value_distribution_file_with_alpha_threshold, write_alpha_value_distribution_file_with_two_alpha_thresholds, compare_background_vs_tumor_and_paired_normaltissue_alpha_value_distribution_files_with_memory_saving_way_for_cancer_detection_V5

method = sys.argv[1] # For example, 'hyper.alpha.samplesetfreq.triple.thresholds.nn0.2.p1,anrange0_0.5.adiff0.5,readfrac+pairedtissuediff0.3-ncplasma0.2' for cancer detection hypermethylation markers, 'hyper.alpha.samplesetfreq.thresholds.n2.p4.minreadfrac+0.1-0.01' for TOO hypermethylation markers
in_file1_background = sys.argv[2]
in_file2_cancer = sys.argv[3]

if 'triple' in method:
    if len(sys.argv)!=6:
        sys.stderr.write('Error: Method that requires triple input files have only two input files!\n  method: %s\n  file1: %s\n  file2: %s\nExit.\n'%(method, in_file1_background, in_file2_cancer))
        sys.exit(-1)
    in_file3_paired_normaltissues = sys.argv[4]
    out_alpha_values_distr_file = sys.argv[5]
else:
    out_alpha_values_distr_file = sys.argv[4]

if 'triple' in method:
    print('Compare three alpha_value_distribution files:\n  method: %s\n  in_file1_background: %s\n  in_file2_cancer: %s\n  in_file3_paired_normaltissues: %s' % (
            method, in_file1_background, in_file2_cancer, in_file3_paired_normaltissues), flush=True)
    ret_marker_2_alpha2freq = compare_background_vs_tumor_and_paired_normaltissue_alpha_value_distribution_files_with_memory_saving_way_for_cancer_detection_V5(
            method, in_file1_background, in_file2_cancer, in_file3_paired_normaltissues)
else:
    print('Compare two alpha_value_distribution files:\n  method: %s\n  in_file1_background: %s\n  in_file2_cancer: %s' % (method, in_file1_background, in_file2_cancer), flush=True)
    if 'readfrac' in method:
        ret_marker_2_alpha2freq = compare_background_vs_cancer_alpha_value_distribution_files_with_memory_saving_way(method, in_file1_background, in_file2_cancer)
    else:
        ret_marker_2_alpha2freq = compare_background_vs_cancer_alpha_value_distribution_files(method, in_file1_background, in_file2_cancer)
print('Write to file\n  out: %s'%out_alpha_values_distr_file, flush=True)
with gzip.open(out_alpha_values_distr_file, 'wt') as fout:
    if 'samplesetfreq' in method:
        if 'triple_V5' in method:
            write_alpha_value_distribution_file_with_two_alpha_thresholds(fout, ret_marker_2_alpha2freq, 'alpha2freqeuncy_is_cumsum')
        else:
            write_alpha_value_distribution_file_with_alpha_threshold(fout, ret_marker_2_alpha2freq, 'alpha2freqeuncy_is_cumsum')
    elif 'enforce_max_output' in method:
        write_alpha_value_distribution_file_with_alpha_threshold(fout, ret_marker_2_alpha2freq, 'enforce_max_output')
    else:
        write_alpha_value_distribution_file_with_alpha_threshold(fout, ret_marker_2_alpha2freq, 'alpha2freqeuncy_is_individual')
print('Done.')
