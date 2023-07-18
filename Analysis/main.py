import matplotlib.pyplot as plt
import pandas as pd
def graph_patents(df, patentnumber):
    cdmapatent = df.loc[df['Pub'] == patentnumber]
    graphlist = cdmapatent['Cited_Date'].to_list()
    #print(graphlist)
    graphlist = [str(x) for x in graphlist]
    graphlist = [int(x[:4]) for x in graphlist]
    n,bins,patches = plt.hist(graphlist,bins = 20, width = 0.75)
    plt.title("Patent Citations for US Patent " +  patentnumber)
    plt.xlabel("Year")
    plt.ylabel("Number of Citations")
    plt.xlim([1988,2023])
    plt.savefig(patentnumber +".png")
    plt.show()
def main():
    df = pd.read_csv("/home/jayden/Code/PatentAnalysis/patent_results1998-2000.csv")
    counts = df['Pub'].value_counts().to_frame()    
    print(counts[:10])
    graph_patents(df, "US-6323846-B1")
main()
