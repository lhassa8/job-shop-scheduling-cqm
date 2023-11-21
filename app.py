# pip install streamlit

# streamlit run app.py



## D-Wave Solver section

from time import time
import warnings

from tabulate import tabulate
import argparse
from dimod import ConstrainedQuadraticModel, Binary, Integer, SampleSet
from dwave.system import LeapHybridCQMSampler

from utils.utils import print_cqm_stats, write_solution_to_file
import utils.plot_schedule as job_plotter
from data import Data


class JSSCQM():
    """Builds and solves a Job Shop Scheduling problem using CQM."""

    def __init__(self):
        self.cqm = None
        self.x = {}
        self.y = {}
        self.makespan = {}
        self.best_sample = {}
        self.solution = {}
        self.completion_time = 0

    def define_cqm_model(self):
        """Define CQM model."""

        self.cqm = ConstrainedQuadraticModel()

    def define_variables(self, data):
        """Define CQM variables.

        Args:
            data: a JSS data class
        """

        # Define make span as an integer variable
        self.makespan = Integer("makespan", lower_bound=0,
                                upper_bound=data.max_makespan)

        # Define integer variable for start time of using machine i for job j
        self.x = {
            (j, i): Integer('x{}_{}'.format(j, i), lower_bound=0,
                            upper_bound=data.max_makespan)
            for j in range(data.num_jobs) for i in range(data.num_machines)}

        # Add binary variable which equals to 1 if job j precedes job k on
        # machine i
        self.y = {(j, k, i): Binary('y{}_{}_{}'.format(j, k, i))
                  for j in range(data.num_jobs)
                  for k in range(data.num_jobs) for i in
                  range(data.num_machines)}

    def define_objective_function(self):
        """Define objective function"""

        self.cqm.set_objective(self.makespan)

    def add_precedence_constraints(self, data):
        """Precedence constraints ensures that all operations of a job are
        executed in the given order.

        Args:
            data: a JSS data class
        """

        for j in range(data.num_jobs):  # job
            for t in range(1, data.num_machines):  # tasks
                machine_curr = data.task_machine[(j, t)]
                machine_prev = data.task_machine[(j, t - 1)]
                self.cqm.add_constraint(self.x[(j, machine_curr)] -
                                        self.x[(j, machine_prev)]
                                        >= data.task_duration[(j, t - 1)],
                                        label='pj{}_m{}'.format(j, t))

    def add_quadratic_overlap_constraint(self, data):
        """Add quadratic constraints to ensure that no two jobs can be scheduled
         on the same machine at the same time.

         Args:
             data: JSS data class
        """

        for j in range(data.num_jobs):
            for k in range(data.num_jobs):
                if j < k:
                    for i in range(data.num_machines):
                        task_k = data.machine_task[(k, i)]
                        task_j = data.machine_task[(j, i)]
                        if data.task_duration[k, task_k] > 0 and\
                                data.task_duration[j, task_j] > 0:
                            self.cqm.add_constraint(
                                self.x[(j, i)] - self.x[(k, i)] + (
                                        data.task_duration[k, task_k] -
                                        data.task_duration[
                                            j, task_j]) * self.y[
                                    (j, k, i)] + 2 * self.y[(j, k, i)] * (
                                        self.x[(k, i)] - self.x[(j, i)]) >=
                                data.task_duration[(k, task_k)],
                                label='OneJobj{}_j{}_m{}'.format(j, k, i))

    def add_makespan_constraint(self, data):
        """Ensures that the make span is at least the largest completion time of
        the last operation of all jobs.

        Args:
            data: JSS data class
        """
        for j in range(data.num_jobs):
            last_machine = data.task_machine[(j, data.num_machines - 1)]
            self.cqm.add_constraint(
                self.makespan - self.x[(j, last_machine)] >=
                data.task_duration[(j, data.num_machines - 1)],
                label='makespan_ctr{}'.format(j))

    def call_cqm_solver(self, time_limit, data):
        """Calls CQM solver.

        Args:
            time_limit: time limit in second
            data: a JSS data class
        """

        sampler = LeapHybridCQMSampler()
        raw_sampleset = sampler.sample_cqm(self.cqm, time_limit=time_limit)
        feasible_sampleset = raw_sampleset.filter(lambda d: d.is_feasible)
        num_feasible = len(feasible_sampleset)
        if num_feasible > 0:
            best_samples = \
                feasible_sampleset.truncate(min(10, num_feasible))
        else:
            warnings.warn("Warning: Did not find feasible solution")
            best_samples = raw_sampleset.truncate(10)

        print(" \n" + "=" * 30 + "BEST SAMPLE SET" + "=" * 30)
        print(best_samples)

        self.best_sample = best_samples.first.sample

        self.solution = {
            (j, i): (data.machine_task[(j, i)],
                     self.best_sample[self.x[(j, i)].variables[0]],
                     data.task_duration[(j, data.machine_task[(j, i)])])
            for i in range(data.num_machines) for j in range(data.num_jobs)}

        self.completion_time = self.best_sample['makespan']


def runApp(instance_name):
    """Modeling and solving Job Shop Scheduling using CQM solver."""

    # Start the timer
    start_time = time()

    

    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description='Job Shop Scheduling Using LeapHybridCQMSampler',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-instance', type=str,
                        help='path to the input instance file; ',
                        default='input/' + instance_name + '.txt')

    parser.add_argument('-tl', type=int,
                        help='time limit in seconds')

    parser.add_argument('-os', type=str,
                        help='path to the output solution file',
                        default='solution.txt')

    parser.add_argument('-op', type=str,
                        help='path to the output plot file',
                        default='schedule.png')

    # Parse input arguments.
    args = parser.parse_args()
    input_file = args.instance
    time_limit = args.tl
    out_plot_file = args.op
    out_sol_file = args.os

    # Define JSS data class
    jss_data = Data(input_file)

    print(" \n" + "=" * 25 + "INPUT SETTINGS" + "=" * 25)
    print(tabulate([["Input Instance", "Time Limit"],
                    [jss_data.instance_name, time_limit]],
                   headers="firstrow"))

    # Read input data
    jss_data.read_input_data()

    # Create an empty JSS CQM model.
    model = JSSCQM()

    # Define CQM model.
    model.define_cqm_model()

    # Define CQM variables.
    model.define_variables(jss_data)

    # Add precedence constraints.
    model.add_precedence_constraints(jss_data)

    # Add constraint to enforce one job only on a machine.
    model.add_quadratic_overlap_constraint(jss_data)

    # Add make span constraints.
    model.add_makespan_constraint(jss_data)

    # Define objective function.
    model.define_objective_function()

    # Print Model statistics
    print_cqm_stats(model.cqm)

    # Finished building the model now time it.
    model_building_time = time() - start_time

    current_time = time()
    # Call cqm solver.
    model.call_cqm_solver(time_limit, jss_data)

    # Finished solving the model now time it.
    solver_time = time() - current_time

    # Print results.
    print(" \n" + "=" * 55 + "SOLUTION RESULTS" + "=" * 55)
    print(tabulate([["Completion Time", "Max Possible Make-Span",
                     "Model Building Time (s)", "Solver Call Time (s)",
                     "Total Runtime (s)"],
                    [model.completion_time, jss_data.max_makespan,
                     model_building_time, solver_time, time() - start_time]],
                   headers="firstrow"))

    # Write solution to a file
    write_solution_to_file(
        jss_data, model.solution, model.completion_time, out_sol_file)

    # Plot solution
    job_plotter.plot_solution(jss_data, model.solution, out_plot_file)

    return(model, model_building_time, solver_time, time() - start_time)



# UI Section


import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
from tabulate import tabulate
from time import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle


# default data
st_model_completion_time = st_max_makespan = st_model_building_time = st_solver_time = st_start_time = 0

data = {
    (0, 0): (0, 0.0, 1),
    (1, 1): (1, 1.0, 2),
    (2, 2): (2, 3.0, 1)
    # This represents a simple scenario with 3 jobs and 3 machines
}

# Base date for the Gantt chart
base_date = datetime(2023, 1, 1)

equipment_names = [
    "Lathe", "Drill Press", "Grinder", "Laser Cutter", "Milling Machine",
    "CNC Machine", "Plasma Cutter", "Water Jet Cutter", "Sheet Metal Bender",
    "Punch Press", "Spot Welder", "MIG Welder", "TIG Welder", "Band Saw",
    "Scroll Saw", "Table Saw", "Jointer", "Planer", "Sanding Machine",
    "Router Table", "Dust Collector", "Air Compressor", "Bench Grinder",
    "Drill Sharpener", "Oscillating Sander", "Chop Saw", "Panel Saw",
    "Edge Bander", "Wood Lathe", "Surface Grinder"
]


#model = runApp()

#data = model.solution

# Save data with pickle
#with open('solution.pkl', 'wb') as f:
#    pickle.dump(data, f)

# Load data from pickle file
with open('solution.pkl', 'rb') as f:
    data = pickle.load(f)

#print("data")
#print(data)

# Function to transform the data for Google Charts
def transform_data(data):
    transformed_data = []
    for (job, machine), (task, start, duration) in data.items():
        if duration > 0:
            machine_name = equipment_names[machine]  # Use the equipment name
            start_millis = start * 60 * 1000  # Convert minutes to milliseconds
            end_millis = (start + duration) * 60 * 1000
            transformed_data.append([machine_name, f"Job {job}", start_millis, end_millis])
    return json.dumps(transformed_data)


chart_data = transform_data(data)


# HTML and JavaScript for the Google Chart with Fira Code font

st.set_page_config(layout="wide")

# Load Fira Code font from Google Fonts
fira_code_font_url = "https://fonts.googleapis.com/css2?family=Fira+Code&display=swap"

# Custom CSS to set Fira Code as the global font
st.markdown(f"""
    <style>
        @import url('{fira_code_font_url}');
        html, body, [class*="st-"] {{
            font-family: 'Fira Code', monospace;
        }}
    </style>
    """, unsafe_allow_html=True)

# Define the options for the dropdown
options = ["instance3_3", "instance5_5", "instance5_8", 
           "instance6_6", "instance8_8", "instance10_10", "instance30_30"]

# Add the select box to the sidebar
instance_name = st.sidebar.selectbox("Select an Instance", options)

def read_lines(filename):
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        second_line = file.readline().strip()  # strip() removes any trailing newline character
    return first_line, second_line


first_line, second_line = read_lines('input/' + instance_name + '.txt')
st.sidebar.markdown(f"""
    <style>
        .big-bold-text {{
            font-size: 24px; /* Adjust size as needed */
            font-weight: bold;
        }}
    </style>
    <p class='big-bold-text'>{first_line}</p>
    <p class='big-bold-text'>{second_line}</p>
    """, unsafe_allow_html=True)



if st.sidebar.button('Run'):
    
    model, model_building_time, solver_time, run_time = runApp(instance_name)
    data = model.solution
    chart_data = transform_data(data)
    
    # Save data with pickle so initial load has values without having to run 
    with open('solution.pkl', 'wb') as f:
        pickle.dump(data, f)

    values = [f"{model_building_time:.2f} sec.", 
                f"{solver_time:.2f} sec.",
                f"{run_time:.2f} sec."]

    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        "Metric": ["Model Building Time (s)", "Solver Call Time (s)", "Total Runtime (s)"],
        "Value": values
    }).set_index("Metric")

    # Custom CSS for table styling
    st.markdown("""
        <style>
            .dataframe th {
                font-family: 'Fira Code', monospace;
                font-size: 18px;
                color: #4F8BF9;  /* Adjust header color as needed */
            }
            .dataframe td {
                font-family: 'Fira Code', monospace;
                font-size: 16px;
                color: #333333;  /* Adjust cell color as needed */
            }
            .stTable {
                width: 800px;  /* Set the width of the table */
            }
        </style>
        """, unsafe_allow_html=True)

    # Display the DataFrame as a table in Streamlit
    st.sidebar.table(results_df)



# Custom CSS to make the sidebar skinnier
st.markdown("""
    <style>
        .css-1d391kg {
            width: 150px;  /* Adjust the width as needed */
        }
    </style>
    """, unsafe_allow_html=True)


# width: 1000,
# height: 1000,

# HTML and JavaScript for the Google Chart with adjusted height
html_string = f"""
<!DOCTYPE html>
<html>
  <head>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code&display=swap" rel="stylesheet">
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <script type="text/javascript">
      google.charts.load("current", {{packages:["timeline"]}});
      google.charts.setOnLoadCallback(drawChart);

      function drawChart() {{
        var container = document.getElementById('timeline');
        var chart = new google.visualization.Timeline(container);
        var dataTable = new google.visualization.DataTable();

        dataTable.addColumn({{ type: 'string', id: 'Machine' }});
        dataTable.addColumn({{ type: 'string', id: 'Job' }});
        dataTable.addColumn({{ type: 'number', id: 'Start' }});
        dataTable.addColumn({{ type: 'number', id: 'End' }});

        
        dataTable.addRows({chart_data});
        

        var options = {{
          width: 1200  
        }};

        chart.draw(dataTable, options);
      }}
    </script>
    <style>
      body {{
        font-family: 'Fira Code', monospace;
      }}
      #chart-container {{
        width: auto;  // Set the width of the container to 100%
        overflow: auto;  // Enable scrolling if the chart is wider than the container
      }}
      #timeline {{
        height: 600px;
        // width: 1200px  
      }}
    </style>
  </head>
  <body>
    <div id="chart-container">
      <div id="timeline"></div>
    </div>
  </body>
</html>
"""


# Streamlit app
st.title("Job Scheduling Dashboard")

# Embed the chart in Streamlit
st.components.v1.html(html_string, height=700, width=1200)


# Assuming 'model', 'model_building_time', 'solver_time', and 'start_time' are already defined

# Create a DataFrame with the results


