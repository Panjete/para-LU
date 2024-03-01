import plotly.graph_objects as go

## Script for generating plots for times for different iterations of OpenMP implementations

a_vanilla = [840, 592]
a1 =      [397, 254, 293]
a2 =      [296, 196, 249]
a3 =      [397, 254, 293]
a4 =      [261, 182, 217]
a5 = [709, 392, 256, 254]

y_vanilla = [1, 2]
y1 = [2, 4, 8]
y2 = [1, 2, 4, 8]

fig = go.Figure(data=[go.Scatter(x=y_vanilla,
                                 y=a_vanilla,
                                 mode='lines', 
                                 name='Vanilla Parallelisation')]
                                 )
fig.add_trace(go.Scatter(x=y1,
                         y=a1,
                         mode='lines',
                         name='Iteration I'))
fig.add_trace(go.Scatter(x=y1,
                         y=a2,
                         mode='lines',
                         name='Iteration II'))
fig.add_trace(go.Scatter(x=y1,
                         y=a3,
                         mode='lines',
                         name='Iteration III'))
fig.add_trace(go.Scatter(x=y1,
                         y=a4,
                         mode='lines',
                         name='Iteration IV'))
fig.add_trace(go.Scatter(x=y2,
                         y=a5,
                         mode='lines',
                         name='Column Major'))
fig.update_layout(title='Time Taken vs #threads for different iterations',scene=dict(aspectmode='data'))
#fig.show()
fig.write_html("openmp_iterations.html")


openmp = [450.248, 249.121, 126.455, 76.328, 70.074]
pthreads = [704.558, 417.096, 239.217, 161.268, 147.128]

openmp_speedup = []
pthread_speedup = []
openmp_efficiency = []
pthread_efficiency = []
for i in range(5):
    openmp_speedup.append(openmp[i]/450.248)
    pthread_speedup.append(pthreads[i]/704.558)
for i in range(5):
    openmp_efficiency.append(openmp_speedup[i]/(i+1))
    pthread_efficiency.append(pthread_speedup[i]/(i+1))

thread_nums = [1, 2, 4, 8, 16]

fig3 = go.Figure(data=[go.Scatter(x=thread_nums,
                                  y=openmp_speedup,
                                    mode='lines', 
                                    name='OpenMP Speed Up')]
                                 )

fig3.add_trace(go.Scatter(x=thread_nums,
                         y=openmp_efficiency,
                         mode='lines',
                         name='OpenMP Efficiency'))

fig3.add_trace(go.Scatter(x=thread_nums,
                         y=pthread_speedup,
                         mode='lines',
                         name='pthreads Speedup'))

fig3.add_trace(go.Scatter(x=thread_nums,
                         y=pthread_efficiency,
                         mode='lines',
                         name='pthread Efficiency'))


fig3.update_layout(title='Efficiencies and Speedups vs #threads for OpenMP and pthreads',scene=dict(aspectmode='data'))
#fig3.show()
fig3.write_html("efficiencies.html")


############

fig2 = go.Figure(data=[go.Scatter(x=thread_nums,
                                  y=openmp,
                                    mode='lines', 
                                    name='OpenMP Times')]
                                 )
fig2.add_trace(go.Scatter(x=thread_nums,
                         y=pthreads,
                         mode='lines',
                         name='pthread Times'))

fig2.add_trace(go.Scatter(x=[1],
                         y=[610.086],
                         mode='markers',
                         name='Sequential Time'))

fig2.update_layout(title='Time Taken vs #threads for OpenMP and pthreads',scene=dict(aspectmode='data'))
fig2.write_html("times.html")


