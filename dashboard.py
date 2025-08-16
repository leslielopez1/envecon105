#import packages
import streamlit as st
import pandas as pd
import openpyxl
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import matplotlib.cm as cm
#loadingdata
@st.cache_data
def load_data():
    url1 = "https://raw.githubusercontent.com/leslielopez1/envecon105/main/yearly_co2_emissions_1000_tonnes%20(1).xlsx"
    CO2_emissions = pd.read_excel(url1)

    url2 = "https://raw.githubusercontent.com/leslielopez1/envecon105/main/gdp_per_capita_yearly_growth.xlsx"
    gdp_growth = pd.read_excel(url2)

    url3 = "https://raw.githubusercontent.com/leslielopez1/envecon105/main/energy_use_per_person.xlsx"
    energy_use = pd.read_excel(url3)

    url4 = "https://raw.githubusercontent.com/leslielopez1/envecon105/main/emdat-country-profiles_mex_2025_08_12.xlsx"
    mex_disaster = pd.read_excel(url4, skiprows=[1])

    url5 = "https://raw.githubusercontent.com/leslielopez1/envecon105/main/cmip6-x0.25_timeseries_tas_timeseries_annual_1950-2014_median_historical_ensemble_all_mean.xlsx"
    mex_temp = pd.read_excel(url5)
    mex_temp = mex_temp.melt(id_vars=['code', 'name'], var_name='Date', value_name='Temperature')

    return CO2_emissions, gdp_growth, energy_use, mex_disaster, mex_temp

CO2_emissions, gdp_growth, energy_use, mex_disaster, mex_temp = load_data()
#creates main section
header = st.container()
intro = st.container()
data_visual = st.container()
data_analysis = st.container()
#summary = st.container() in case we need one last part

#define the content of each section below: 
with header: 
    st.title("Mexico's CO\u2082 Emissions and Associations over Time")

with intro:
    st.header("Motivation")
    st.markdown('''Mexico in the early 2000s signed the Kyoto Protocol, committing to control their greenhouse gas emissions. 
    Since then, Mexico has introduced legislation aiming to further counter climate change, such as the Energy Transition Law. 
    Their actions have served as an example for other developing countries.
    Understanding how Mexico's economy heavily relies on fossil fuels for energy production, transportation, and operations, this project aims to study the change in emissions and other associations within the enviromental sector.
    Thanks to their enviormentally-conscious president, Claudia Sheinbaum, Mexico has been taking bold strides in reducing their emissions and carbon footprint. 
    Historically, they have been responsible for 1.6% of the world's CO2 emissions, largely from industrial fuels. This has become largely because of the market shift from an agrarian economy to more industrialized which in turn emissions went up as well.
    With active participation in the Kyoto Protocol and Paris Agreement, they have commited to reducing their GHG emissions- from 25% to, recently updated, 35% in 2022. 
    Rising CO2 emissions create harmful situations that impact the environment and natural processes including global warming and increased sea levels. 
    Specifically, Mexico can experience significant deforestiation, desertification, and drought as rain and climate patterns are altered.
    
Citations:
- “Mexico.” UNDP Climate Promise, climatepromise.undp.org/what-we-do/where-we-work/mexico.
- “Mexico and Greenhouse Gas Emissions: EBSCO.” EBSCO Information Services, Inc. | Www.Ebsco.Com, www.ebsco.com/research-starters/politics-and-government/mexico-and-greenhouse-gas-emissions.
''')
#Main Research Qs
    st.header("Main Research Questions")
    st.markdown('''1) How has Mexico's CO2 emissions changed over time? And how
        does Mexico compare to other countries (the rest of the world)?''')
    st.markdown('''2) Are CO2 emissions, temperature, and natural disasters in Mexico associated?''')
#Context:
    st.header("Context")
    st.markdown('''Comprised of multiple sources, these datasets span over different time periods and will be used to explore how CO2 emissions relate to economic, environmental, and climate indicators.

Measurements:
- **CO2 Emissions**: reflecting the average annual emissions produced by one person in each country
- **GDP per capita**: year over year percentage change in economic output per person- how quicly the average wealth level grows
- **Energy per person**: proxy for overall energy consumption, amount of energy used by an individual annually
- **Disasters**: yearly rate of natural disaster events
- **Temperature**: yearly mean temperature (in celsusis)''')
#Limitations:
    st.header("Limitations")
    st.markdown('''Including CO2 emissions, the following data is a collection of different datasets which may be influenced by CO2 emissions or associated. 
    This data allows us to examine trends within each category and compare them over the years. 
    Additionally,  by exploring the trends we can see how enviormental and economic trends are correlated.''')
    st.markdown('''That being said this data measures only the countries and years which they reported to various agencies, so it is not an all true  global analysis. 
    With incomplete data, we can't draw definitive conclusions about cause and effect relationships between different categories. 
    As the data is generalized to represent the countries as a whole, we also can not determine specific regional impacts or short-term trends.''')


with data_visual: 
    st.header("What is the Data?")

    st.header("Data Visualization")
    st.subheader("1. Country CO2 Emissions per Year (1751-2014)")
    st.markdown("""This graph shows that Mexico is one of the countries that produces the lowest amount of CO2 emissions compared to the United States, which has dominated as the largest CO2 emission producing country until recently. 
    Mexico's emissions have risen slightly since the year 2000.""")
    filtered = data_long[data_long['Indicator'] == 'Emissions']
    filtered_mex = data_long[
        (data_long['Indicator'] == 'Emissions') & (data_long['Country'] == 'Mexico')
    ]

    summary_mex = filtered_mex.groupby('Year', as_index=False)['Value'].sum()
    summary_mex.rename(columns={'Value': 'Emissions'}, inplace=True)

    summary = filtered.groupby(['Year', 'Country'], as_index=False, observed=True)['Value'].sum()
    summary.rename(columns={'Value': 'Emissions'}, inplace=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for country, group_data in summary.groupby('Country', observed=True):
        ax.plot(group_data['Year'], group_data['Emissions'], color='black', alpha=0.4)

    ax.plot(summary_mex['Year'], summary_mex['Emissions'], label='Mexico', color='blue')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Emissions (Metric Tonnes)', fontsize=12)
    ax.set_title(r"Country $CO_2$ Emissions per year (1751-2014)", fontsize=17)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(title='Country', loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.figtext(0.05,0.01, "Limited to reporting countries", fontsize=12)

    # Display in Streamlit
    st.pyplot(fig)

    st.subheader("2. Top 10 Emissions-producing Countries in 2010 (1900-2014)")
    st.markdown("""This graph shows that among the top 10 emission producing countries in 2010, Mexico lays relatively low compared to the others, ranking #13 globally.""")
    st.subheader("3. Tile Plot: Top 10 CO2 Emission-producing Countries")
    st.markdown("""This tile plot shows the change over time until 2014 in emisssions produced within the top 10 countries compared to Mexico. 
    As a developing country, Mexico started to produce more emissions in the 20th century and is now working to not increase their levels.""") 
    st.subheader("4. Faceted Plot: Indicator Comparison of Mexico and the Rest of the World")
    st.markdown("""This type of data visualization method presents how the different indicators in the dataset change through time and how they compare with each other. 
    These plots demonstrates that each type of data spans a different time span.""")
    st.subheader("5) Scatter Plot: CO2 Emissions and Temperature")
    st.markdown("""These distinct scatter plots show that there are similar patterns of CO2 emission levels and average annual temperatures. 
    There seems to be an association between high CO2 emissions and the rise of temperatures in Mexico.""")


with data_analysis:
    st.header("Data Analysis")
    st.subheader("1. Calculate the Mean and SD for emissions and temperature for Mexico")
    st.subheader("2. Calculate the correlatation coefficient for emissions and temperature")
    st.markdown("""The correlation coefficient measures the strenght of a linear relationship between two variables. 
    In this case, a correlatation coefficient of about 0.93 indicates a strong correlation between CO2 emissions and temperature. 
    However, a linear relationship might not be the best way to capture the relationship, since a correlation does not impy causation.""")
    st.subheader("3. Scaled scatter plot showing the correlation between emissions and temperature in Mexico")
