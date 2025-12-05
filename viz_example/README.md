Data visiualization additions

trace_exec_training_list.py is modified to add three columns to the results.csv file that is generated:
TblSize, HistLen, and LayerSize. These are used to hold the values for those parameters for the purposes
of chart generation. The results.csv file also has a timestamp appended to its filename for uniqueness.

The script can take one of three new command line arguments to add value tags to the csv file:

--table_size <size> 
--hist_len <length>
--layer_size <size>

These can be used by the data_viz.py script to group bars according to parameter.

data_viz.py

data_viz.py generates three charts from a directory containing one or more results csv files. Charts compare
MPKI, MR, and IPC. The script takes the following command line arguments:

--sub <"sub">   Optional subtitle string to be included below the main chart title. Intended to be used to note the 
                fixed parameter among the compared groups.
--run           Use the "Run" column instead of "Workload" as the primary category (legend and single-level X-axis).
                If the trace_dir given to trace_exec_training_list.py contains subdirectories containing traces each 
                suddirectory is identified as a unique workload in the workload column. If the trace_dir contains only 
                traces, the workload column will be the same for all traces, but each entry in the run column is unique
                to the trace. In this case, using --run will generate bars from the individual traces.
--hist          Group bars by the "HistLen" column.
--table         Group bars by the "TblSize" column.
--layer         Group bars by the "LayerSize" column.

The example charts were generated with the following workflow:

1. Copy desired traces to directory viz_traces
2. Make the perceptron simulator with table size set to 256
3. Run: 
    python3 scripts/trace_exec_training_list.py --trace_dir viz_traces --simulator ./cbp_perceptron --table_size 256 --results_dir ../results_tbl
4. Repeat steps 1 and 2 changing the table size and tag 512 and again to 1024
5. Go through the above steps changing history length and the tag using --hist_len for values of 30 and 62, setting   
   result_dir to results_hist

6. Run: 
    python3 scripts/data_viz.py --run --table --sub "History Length = 62" results_tbl
7. Run: 
    python3 scripts/data_viz.py --run --hist --sub "Table Size = 1024" results_hist
8. Enjoy charts found in the charts directory of the respective results directories