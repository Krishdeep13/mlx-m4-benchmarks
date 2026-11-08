cd "$(dirname "$0")"
caffeinate -i -s python3.15 item0_env.py > item0.log 2>&1
caffeinate -i -s python3.15 item1_timing_lies.py > item1.log 2>&1
caffeinate -i -s python3.15 item2_single_vs_median.py > item2.log 2>&1
caffeinate -i -s python3.15 item3_fusion.py > item3.log 2>&1
caffeinate -i -s python3.15 item5_ksweep.py > item5.log 2>&1
caffeinate -i -s python3.15 item5b_staircase.py > item5b.log 2>&1
caffeinate -i -s python3.15 item6_compile_breaks.py > item6_compile_breaks.log 2>&1
caffeinate -i -s python3.15 item7_fp16_and_tiny.py > item7_fp16_and_tiny.log 2>&1
caffeinate -i -s python3.15 item9_fusion_suite.py > item9.log 2>&1
caffeinate -i -s python3.15 item10_extra.py > item10.log 2>&1
echo "ALL DONE"
