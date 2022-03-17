import matplotlib.pyplot as plt
from matplotlib import image
import pandas as pd
from matplotlib import colors
from IPython.display import display


def plot_image(data):
    #fig = plt.figure(figsize=(15, 7))
    plt.title("Adversial examples")
    plt.imshow(data,cmap=plt.cm.gray, interpolation='nearest')
    plt.show() 

def plot_images(data):
    # plot the images: each image is 28x28 pixels
    # set up the figure
    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    #for now, only show top 50
    if(data.shape[0]<=50):
        total_number = data.shape[0]
    else:
        total_number = 50
    for i in range(total_number):
        #ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
        ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
        #here the interpolation controls "anti-aliasing". For our case this should be strictly "none" or "nearest" because
        #we dont need antiliasing since most of our decisions on selecting hyperparameters are based on visual clues
        #ax.imshow(data[i].reshape((32,32,3)), interpolation='nearest')
        ax.imshow(data[i].reshape((28,28)),cmap=plt.cm.gray, interpolation='nearest')
        #ax.text(0, 7, i, color='red', weight="bold")
        #ax.imshow(data[i].reshape((28,28)),cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

def plotTable(data):   
    def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
        rng = M - m
        norm = colors.Normalize(m - (rng * low),
                                M + (rng * high))
        normed = norm(s.values)
        c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
        return ['background-color: %s' % color for color in c]

    data.style.apply(background_gradient,
                cmap='PuBu',
                m=data.min().min(),
                M=data.max().max(),
                low=0,
                high=0.2)

def plot_data_points(data_frame):
    df = pd.DataFrame(data_frame)
    pd.set_option("display.precision", 3)
    df.columns= ["Float", "1" ,"2", "4", "8", "12", "16"]
    df.index = ["Float", "1", "2", "4", "8", "12", "16"]
    #visualize_data.plotTable(df)
    # generate some example data
    def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
        rng = M - m
        norm = colors.Normalize(m - (rng * low),
                                M + (rng * high))
        normed = norm(s.values)
        c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
        return ['background-color: %s' % color for color in c]

    display(df.style.apply(background_gradient,
                cmap='PuBu',
                m=df.min().min(),
                M=df.max().max(),
                low=0,
                high=0.2))

def plot_data_points_with_average(data_frame):
    df = pd.DataFrame(data_frame)
    pd.set_option("display.precision", 3)
    df.columns= ["Float", "1" ,"2", "4", "8", "12", "16", "Average"]
    df.index = ["Float", "1", "2", "4", "8", "12", "16"]
    #visualize_data.plotTable(df)
    # generate some example data
    def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
        rng = M - m
        norm = colors.Normalize(m - (rng * low),
                                M + (rng * high))
        normed = norm(s.values)
        c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
        return ['background-color: %s' % color for color in c]

    display(df.style.apply(background_gradient,
                cmap='PuBu',
                m=df.min().min(),
                M=df.max().max(),
                low=0,
                high=0.2))


def plot_data_points_with_average_stylized(data_frame):
    df = pd.DataFrame(data_frame)
    pd.set_option("display.precision", 3)
    df.columns= ["FP", "1" ,"2", "4", "8", "12", "16", "Average"]
    df.index = ["FP", "1", "2", "4", "8", "12", "16"]
    #visualize_data.plotTable(df)
    # generate some example data
    def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
        rng = M - m
        norm = colors.Normalize(m - (rng * low),
                                M + (rng * high))
        normed = norm(s.values)
        c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
        return ['background-color: %s' % color for color in c]

    th_props = [
                ('font-weight', 'bold'),
                ('background-color', '#f7f7f9')
                ]

    # Set table styles
    styles = [
        dict(selector="th", props=th_props)
        ]

    display(df.style.apply(background_gradient,
                cmap='PuBu',
                m=df.min().min(),
                M=df.max().max(),
                low=0,
                high=0.2).set_table_styles(styles))


def plot_data_points_with_modelname(data_frame):
    df = pd.DataFrame(data_frame)
    pd.set_option("display.precision", 3)    
    df.style.set_caption("This is Analytics Vidhya Blog")    
    columns= ["Float", "1" ,"2", "4", "8", "12", "16", "Average"]
    columns= list(zip(['HEADER1'] * 8, columns)) 
    df.columns =  pd.MultiIndex.from_tuples(columns)
    indexx = ["Float", "1", "2", "4", "8", "12", "16"]
    indexx= list(zip(['HEADER2'] * 8, indexx)) 
    df.index =  pd.MultiIndex.from_tuples(indexx)
    #visualize_data.plotTable(df)
    # generate some example data
    def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
        rng = M - m
        norm = colors.Normalize(m - (rng * low),
                                M + (rng * high))
        normed = norm(s.values)
        c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
        return ['background-color: %s' % color for color in c]   

    td_props = [
                ('font-weight', 'bold'),
                ('background-color', '#ffffff')


                ]
    th_props = [
                ('font-weight', 'bold'),
                ('background-color', '#f7f7f9')
                ]
    # Set table styles
    styles = [
        dict(selector="th", props=th_props),  
        dict(selector="HEADER2", props=td_props)    
        ]
    df.style.format({"HEADER2": lambda x: "±{:.2f}".format(abs(x))})

    display(df.style.apply(background_gradient,
                cmap='PuBu',
                m=df.min().min(),
                M=df.max().max(),
                low=0,
                high=0.2).set_table_styles(styles))