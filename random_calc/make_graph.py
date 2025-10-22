import numpy as numpy
import matplotlib.pyplot as plt 
import pandas as pd 

def read_file(filename):
    df = pd.read_csv(filename)
    return df

def plotting_ionic_calc(df):
    plt.figure(figsize=(10,6))
    plt.plot(df['depth'], df['lat range '], 'b-o', linewidth=2, markersize=6, label='Lateral Range')
    plt.plot(df['depth'], df['lat straddle'], 'r-s', linewidth=2, markersize=6, label='Lateral Straggling')
    # plt.plot(df['depth'], df['Longitudinal range'], 'g-^', linewidth=2, markersize=6, label='Longitudinal Range')
    plt.plot(df['depth'], df['long straddle'], 'm-d', linewidth=2, markersize=6, label='Longitudinal Straggling')
    plt.xlabel('Depth (micrometers)', fontsize=12)
    plt.ylabel('Distance (micrometers)', fontsize=12)
    plt.title('Ion calc Parameters vs Depth', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()
def plotting_stopping_table(df):
    df.columns = ['Energy_keV', 'dE_dx_Electronic', 'dE_dx_Nuclear', 
              'Projected_Range_um', 'Longitudinal_Straggling_um', 'Lateral_Straggling_um']
    plt.figure(figsize = (12,8))
    # plt.plot(df['Energy_keV'], df['Projected_Range_um'], 'g-^', linewidth=2, markersize=3, label='Projected Range')
    plt.plot(df['Energy_keV'], df['Longitudinal_Straggling_um'], 'm-d', linewidth=2, markersize=3, label='Longitudinal Straggling')  
    plt.plot(df['Energy_keV'], df['Lateral_Straggling_um'], 'r-s', linewidth=2, markersize=3, label='Lateral Straggling')
    plt.xlabel('Beam Energy (keV)', fontsize=12)
    plt.ylabel('Distance (micrometers)', fontsize=12)   
    plt.title('Ions Stats Parameters vs Energy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, df['Energy_keV'].max())
    plt.ylim(0, None)

    plt.tight_layout()
    plt.show()
ionic_calc = read_file('ionic_calc_10sep.csv')
stopping_table = read_file('stopping_table.csv')
# print(stopping_table['Energy_keV'])
plotting_ionic_calc(ionic_calc)
# plotting_stopping_table(stopping_table)