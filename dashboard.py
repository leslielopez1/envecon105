#import packages
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

#loadingdata
@st.cache_data
def load_data():
    url1 = "https://raw.githubusercontent.com/leslielopez1/envecon105/main/yearly_co2_emissions_1000_tonnes%20(1).xlsx"
    co2_emissions = pd.read_excel(url1)

    url2 = "https://raw.githubusercontent.com/leslielopez1/envecon105/main/gdp_per_capita_yearly_growth.xlsx"
    gdp_growth = pd.read_excel(url2)

    url3 = "https://raw.githubusercontent.com/leslielopez1/envecon105/main/energy_use_per_person.xlsx"
    energy_use = pd.read_excel(url3)

    url4 = "https://raw.githubusercontent.com/leslielopez1/envecon105/main/emdat-country-profiles_mex_2025_08_12.xlsx"
    mex_disaster = pd.read_excel(url4, skiprows=[1])

    url5 = "https://raw.githubusercontent.com/leslielopez1/envecon105/main/cmip6-x0.25_timeseries_tas_timeseries_annual_1950-2014_median_historical_ensemble_all_mean.xlsx"
    mex_temp = pd.read_excel(url5)
    mex_temp = mex_temp.melt(id_vars=['code', 'name'], var_name='Date', value_name='Temperature')
    return {
        "co2_emissions": co2_emissions,
        "gdp_growth": gdp_growth,
        "energy_use": energy_use,
        "mex_disaster": mex_disaster,
        "mex_temp": mex_temp
    }
#creates a dictionary with key=names and value=dataframe
data = load_data()

#creates main section
header = st.container()
intro = st.container()
data_visual = st.container()
data_analysis = st.container()
#summary = st.container() in case we need one last part

#define the content of each section below: 
with header: 
    st.title("Exploring Mexico's CO\u2082 Emissions and Associations over Time")

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
- **Temperature**: yearly mean temperature (in celsius)''')
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
    st.markdown("""| Data                                         | Time span   | Source    | Original Source | Description                                                                                                                                         | Citation |
|----------------------------------------------|-------------|-----------|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| CO2 emissions                                | 1751–2014   | Gapminder | [CDIAC](https://cdiac.ess-dive.lbl.gov/) | CO2 emissions in tonnes or metric tons (≈ 2,204.6 pounds) per person by country                                                                     | NA       |
| GDP per capita (percent yearly growth)       | 1801–2019   | Gapminder | [World Bank](https://www.worldbank.org/) | Gross Domestic Product (overall measure of a nation's economic health) per person by country                                                       | NA       |
| Energy use per person                        | 1960–2015   | Gapminder | [World Bank](https://www.worldbank.org/) | Use of primary energy before transformation to other end-use fuels, by country                                                                      | NA       |
| Mexico Natural Disasters                         | 2000–2025  | Humanitarian Data Exchange  | [EM-DAT](https://www.emdat.be/)   | Frequency and impact of natural disasters in Mexico, includes Droughts, Floods, Freezes, Severe Storms, Tropical Cyclones, Wildfires | EM-DAT, CRED / UCLouvain, 2025, Brussels, Belgium, www.emdat.be|
| Temperature                                  | 1950–2014   |  Climate Change Knowledge Portal        | [World Bank](https://climateknowledgeportal.worldbank.org/)                | Mexico National yearly average temperature (in Celsius) from 1950 to 2014              | “World Bank Climate Change Knowledge Portal.” Worldbank.org, 2021, [climateknowledgeportal.worldbank.org/](https://climateknowledgeportal.worldbank.org/). Accessed 10 Aug. 2025.|
""")
    
    st.header("Data Import")
    st.markdown('''In the following data tables, data was modified to make it easier to visualize by converting the wide format into a long or "narrow" format 
    using pivot tables.''')
    st.subheader("Yearly CO2 Emissions")
    CO2_emissions = data["co2_emissions"]
    CO2_emissions_mod = pd.melt(CO2_emissions, id_vars = ['country'],
                        var_name = 'Year', value_name = 'Emissions')
    #converting year to numeric
    CO2_emissions_mod['Year'] = pd.to_numeric(CO2_emissions_mod['Year'], errors = 'coerce')
    CO2_emissions_mod = CO2_emissions_mod.rename(columns={"country": "Country"})
    #add label variable
    CO2_emissions_mod['Label'] = "CO2 Emissions (Metric Tons)"
    st.dataframe(CO2_emissions_mod)

    st.subheader("Yearly Growth in GDP per Capita")
    gdp_growth = data["gdp_growth"]
    gdp_growth_mod = gdp_growth.melt(id_vars=["country"], var_name="year", value_name="GDP")
    #converting year to numeric
    gdp_growth_mod["year"] = pd.to_numeric(gdp_growth_mod["year"], errors='coerce')
    #renaming
    gdp_growth_mod = gdp_growth_mod.rename(columns={ 'country': 'Country'})
    #label
    gdp_growth_mod["Label"]= "GDP Growth/Capita (%)"
    st.dataframe(gdp_growth_mod)

    st.subheader("Energy Use per person")
    energy_use = data["energy_use"]
    energy_use_mod = pd.melt(energy_use, id_vars = ['country'], var_name = 'Year', value_name = 'Energy')
    energy_use_mod["Year"] = pd.to_numeric(energy_use_mod["Year"], errors = 'coerce')
    energy_use_mod = energy_use_mod.rename(columns={"country": "Country"})
    energy_use_mod["Label"] = "Energy Use (kg, oil-eq./capita)"
    st.dataframe(energy_use_mod)
    
    st.subheader("Mexico Data: Natural Disasters and Annual Temperatures")
    st.subheader("Disasters")
    st.markdown('''The original table had additional data that is not relevant for this study including the Total Affected, Total Damage, and other variables. The columns of interest are Year and Total Events.
    The following was done to clean the data.
- Find the total disasters occuring in a year and label as Value
- Add a Country variable
- Add an Indicator variable
- Add a Label variable''')
    mex_disaster = data["mex_disaster"]
    #find the total number of disasters occuring each year
    disaster_type = mex_disaster[["Year", "Country", "Disaster Type", "Total Events"]].groupby("Year")["Total Events"].sum().reset_index()
    disaster_type['Year'] = pd.to_numeric(disaster_type['Year'], errors='coerce')
    #rename 'total event' to 'value'
    mex_disaster_mod = disaster_type.rename(columns={"Total Events": "Value"})
    #create 'label', 'indicator', and 'country' labels and reorder columns
    mex_disaster_mod["Label"] = "Number of Disasters"
    mex_disaster_mod.insert(1, "Indicator", "Disasters")
    mex_disaster_mod.insert(1, 'Country', 'Mexico')
    st.dataframe(mex_disaster_mod)
    
    st.subheader("Temperature")
    st.markdown('''The following was done to clean the data.
- Date & renamed to Year
- Add a country variable
- Add a Indicator variable
- Add a Label Variable''') 
    mex_temp = data["mex_temp"]
    mex_temp['Date'] = mex_temp['Date'].astype(str).str[:4]
    mex_temp['Country'] = 'Mexico'
    #rename 'date' to 'year' and convert to numeric
    mex_temp = mex_temp.rename(columns={'Date': 'Year'})
    mex_temp['Year'] = pd.to_numeric(mex_temp['Year'], errors='coerce')
    #rename 'temperature' to 'value
    mex_temp = mex_temp.rename(columns={'Temperature': 'Value'})
    #create 'indicator' variable
    mex_temp['Indicator'] = 'Temperature'
    #create 'label' variable
    mex_temp['Label'] = 'Temperature (Celsius)'
    #reorder columns
    mex_temp_mod = mex_temp[['Year', 'Country', 'Indicator', 'Value', 'Label']]
    st.dataframe(mex_temp_mod)

    st.subheader("Joined and Clean Data")
    st.markdown('''Datasets were joined using the pandas .merge function and .melt function in order to combine the tables using common variables such as Country, Year, and Label. 
    The Mexico-based datasets were added to the end of the final dataframe using the pandas .concat function.''')
    #joining through .merge()
    #column names are consistent
    for df in [CO2_emissions_mod, gdp_growth_mod, energy_use_mod]:
        if 'year' in df.columns:
            df.rename(columns={'year': 'Year'}, inplace=True)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')  # make numeric if not already

    #CO2_emissions with GDP_growth
    data_join = pd.merge(CO2_emissions_mod, gdp_growth_mod,
        on=["Country", "Year", "Label"], how="outer")
    #join with energy_use
    data_join = pd.merge(data_join, energy_use_mod,
    on=["Country", "Year", "Label"], how="outer")
    #New variable "indicator"
    data_long = pd.melt(data_join, id_vars=['Country', 'Year', 'Label'], var_name='Indicator', value_name='Value')
    
    data_long = data_long.sort_values(by=['Country', 'Year']).reset_index(drop=True)
    mex_temp_mod = mex_temp_mod.sort_values(by=['Country', 'Year']).reset_index(drop=True)

    #joining Mexico data through .concat()
    data_long = pd.concat([data_long, mex_temp_mod, mex_disaster_mod], ignore_index=True)
    data_long["Country"] = data_long["Country"].astype("category")
    #adding 'region' variable seperate Mexico from other countires
    data_long['Region'] = data_long['Country'].apply(lambda x: "Mexico" if x == "Mexico" else "Rest of the World")
    data_long = data_long.dropna().sort_values(by=['Country', 'Year']).reset_index(drop=True)
    st.dataframe(data_long)

    
    st.header("Data Visualization")
    st.subheader("1. Country CO2 Emissions per Year (1751-2014)")
    st.markdown("""This graph shows that Mexico is one of the countries that produces the lowest amount of CO2 emissions compared to the United States, which has dominated as the largest CO2 emission producing country until recently. 
    Mexico's emissions have risen slightly since the year 2000.""")
    #code for 1st graph
    filtered = data_long[data_long['Indicator'] == 'Emissions']
    filtered_mex = filtered[filtered['Country'] == 'Mexico']

    summary_mex = filtered_mex.groupby('Year', as_index=False)['Value'].sum()
    summary = filtered.groupby(['Year', 'Country'], as_index=False)['Value'].sum()

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for country, group_data in summary.groupby('Country'):
        ax1.plot(group_data['Year'], group_data['Value'], color='black', alpha=0.3)
    ax1.plot(summary_mex['Year'], summary_mex['Value'], label='Mexico', color='blue')
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Emissions (Metric Tonnes)")
    ax1.set_title("Country CO₂ Emissions per year (1751–2014)")
    ax1.legend(title='Country', loc="center left", bbox_to_anchor=(1, 0.5))
    st.pyplot(fig1)

    
    st.subheader("2. Top 10 Emissions-producing Countries in 2010 (1900-2014)")
    st.markdown("""This graph shows that among the top 10 emission producing countries in 2010, Mexico lays relatively low compared to the others, ranking #13 globally.""")
    #code for second graph
    filtered_2010 = filtered[(filtered['Year'] == 2010) & (filtered['Label'] == 'CO2 Emissions (Metric Tons)')].copy()
    filtered_2010['rank'] = filtered_2010['Value'].rank(method='dense', ascending=False)
    top10 = filtered_2010[filtered_2010['rank'] <= 10]['Country'].tolist()
    top10.append('Mexico')
    top10_countries = list(set(top10))

    filtered_top10 = data_long[
        (data_long['Country'].isin(top10_countries)) &
        (data_long['Indicator'] == 'Emissions') &
        (data_long['Year'] > 1900)
    ]

    colors = cm.viridis(np.linspace(0, 1, len(top10_countries)))
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    for color, country in zip(colors, top10_countries):
        country_data = filtered_top10[filtered_top10['Country'] == country]
        ax2.plot(country_data['Year'], country_data['Value'], label=country, color=color)

    plt.title('Ordered by Emissions Produced in 2010', fontsize=12)
    plt.suptitle('Top 10 Emissions-producing Countries in 2010 (1900–2014)', fontsize=16)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Emissions (Metric Tons)", fontsize=12)
    ax2.legend(title='Country', fontsize=8, title_fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig2.tight_layout()
    st.pyplot(fig2)

    
    st.subheader("3. Tile Plot: Top 10 CO2 Emission-producing Countries")
    st.markdown("""This tile plot shows the change over time until 2014 in emisssions produced within the top 10 countries compared to Mexico. 
    As a developing country, Mexico started to produce more emissions in the 20th century and is now working to not increase their levels.""") 
    #code for third graph
    filtered = data_long[
        (data_long['Country'].isin(top10_countries)) &
        (data_long['Indicator'] == 'Emissions') &
        (data_long['Year'] >= 1900)
    ].copy()

    emissions_2014 = filtered[filtered['Year'] == 2014][['Country', 'Value']]
    country_order = emissions_2014.sort_values('Value', ascending=False)['Country']
    filtered['Country'] = pd.Categorical(filtered['Country'], categories=country_order, ordered=True)
    filtered['log_value'] = np.log(filtered['Value'])

    heatmap_data = filtered.pivot(index='Country', columns='Year', values='log_value')
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='viridis',
                cbar_kws={'location': 'bottom', 'pad': 0.2, 'shrink': 0.5}, ax=ax3)
    ax3.set_xlabel("Ln(CO₂ Emissions (Metric Tons))")
    ax3.set_ylabel("")
    plt.suptitle("Top 10 CO₂ Emission-producing Countries vs Mexico", fontsize=16)
    plt.title("Ordered by Emissions Produced in 2014", fontsize=12)
    st.pyplot(fig3)

    st.subheader("4. Faceted Plot: Indicator Comparison of Mexico and the Rest of the World")
    st.markdown("""This type of data visualization method presents how the different indicators in the dataset change through time and how they compare with each other. 
    These plots demonstrates that each type of data spans a different time span.""")
    #code for 4th graph
    mex_indi_filtered = data_long[~data_long['Indicator'].isin(['Disasters', 'Temperature'])]
    grid = sns.FacetGrid(mex_indi_filtered, row="Indicator", col="Region",
                         margin_titles=True, sharex=True, sharey=False,
                         height=3, aspect=1.5)
    grid.map_dataframe(sns.lineplot, x='Year', y='Value', legend=False)
    grid.set_axis_labels("Year", "Indicator Value")
    grid.set_titles(col_template="{col_name}", row_template="{row_name}")
    st.pyplot(grid.fig)

    st.subheader("5) Scatter Plot: CO2 Emissions and Temperature")
    st.markdown("""These distinct scatter plots show that there are similar patterns of CO2 emission levels and average annual temperatures. 
    There seems to be an association between high CO2 emissions and the rise of temperatures in Mexico.""")
    #code for 5th graph
    #Filter data to be between 1980 and 2014
    CO2_temp_mex_facet = data_long[
    (data_long['Country'] == 'Mexico') &
    (data_long['Year'] >= 1980) & (data_long['Year'] <= 2014) &
    (data_long['Indicator'].isin(['Emissions', 'Temperature']))
    ].drop(columns="Label").copy()

    CO2_temp_mex_facet['Indicator'] = CO2_temp_mex_facet['Indicator'].replace({
    'Emissions': 'CO2 Emissions (Metric Tons)',
    'Temperature': 'Temperature (Celsius)'
    })
    def regression_plot(data, **kwargs):
        x = data['Year']
        y = data['Value']
        plt.scatter(x, y, s=15, color='black')
        coeffs = np.polyfit(x, y, deg=1)  # slope & intercept
        y_pred = np.polyval(coeffs, x)
        plt.plot(x, y_pred, color='blue', linewidth=2)


    gr = sns.FacetGrid(CO2_temp_mex_facet, row='Indicator',
                      sharex=True, sharey=False, height=4, aspect=2)
    gr.map_dataframe(regression_plot)
    gr.set_titles(row_template="{row_name}", size=14)
    gr.set_axis_labels("Year", "")
    st.pyplot(gr.fig)


with data_analysis:
    st.header("Data Analysis")
    st.subheader("1. Calculate the Mean and SD for emissions and temperature for Mexico")
    #Make each indicator their own variable by making a wider table
    wide_mex = CO2_temp_mex_facet.pivot(index='Year', columns='Indicator', values='Value')
    #adjust column names to resemble case study
    wide_mex = wide_mex.rename(columns={'Temperature (Celsius)': 'Temperature',
                                  'CO2 Emissions (Metric Tons)': 'Emissions'})

    #show mean and standard deviation for each indicator
    summary = wide_mex.agg(['mean', 'std'])
    st.table(summary)

    st.subheader("2. Calculate the correlatation coefficient for emissions and temperature")
    st.markdown("""The correlation coefficient measures the strenght of a linear relationship between two variables. 
    In this case, a correlatation coefficient of about 0.93 indicates a strong correlation between CO2 emissions and temperature. 
    However, a linear relationship might not be the best way to capture the relationship, since a correlation does not imply causation.""")
    #plot temperature vs. emissions to show the relationship between them
    fig, ax = plt.subplots(figsize=(8,6))

    x = CO2_temp_mex_facet[CO2_temp_mex_facet['Indicator'] == 'CO2 Emissions (Metric Tons)']
    y = CO2_temp_mex_facet[CO2_temp_mex_facet['Indicator'] == 'Temperature (Celsius)']

    ax.scatter(x['Value'], y['Value'], s = 10, color = 'black')

    #plot linear regression line
    m, b = np.polyfit(x["Value"], y["Value"], 1)
    ax.plot(x["Value"], m * (x["Value"]) + b, color='blue', linewidth=2)

    ax.set_xlabel('CO2 Emissions (Metric Tons)', fontsize=12)
    ax.set_ylabel('Temperature (Celsius)', fontsize=12)
    ax.set_title("Mexico CO₂ Emissions and Temperature (1980–2014)", fontsize=16, pad=10)
    st.pyplot(fig)

    #calculate correlation coefficient
    st.write('')
    r = np.corrcoef(x["Value"], y["Value"])[0, 1]
    st.write("Correlation coefficient:", r) #implies high positive correlation

    st.subheader("3. Scaled scatter plot showing the correlation between emissions and temperature in Mexico")
    scaled_mex = wide_mex.copy()
    scaled_mex['Emissions_scaled'] = (scaled_mex['Emissions'] - scaled_mex['Emissions'].mean()) / scaled_mex['Emissions'].std()
    scaled_mex['Temperature_scaled'] = (scaled_mex['Temperature'] - scaled_mex['Temperature'].mean()) / scaled_mex['Temperature'].std()
    #plot graph
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(scaled_mex, x="Emissions_scaled", y="Temperature_scaled",
            scatter_kws={"color": "black", "s": 15},
            line_kws={"color": "blue", "linewidth": 2}, ci=None, ax=ax)

    ax.set_title("Mexico CO₂ Emissions and Temperature (1980–2014)", fontsize=16)
    ax.set_xlabel("Scaled Emissions (Metric Tonnes)", fontsize=14)
    ax.set_ylabel("Scaled Temperature (Celsius)", fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    st.pyplot(fig)

#Interactive graph showing correlation coefficient
st.subheader("Interactive Correlation Explorer")
st.markdown("The graph below describes the correlation between Mexico's CO₂ emissions and a specific indicator.")
CO2_indic_mex_facet = data_long[(data_long['Country'] == 'Mexico') & (data_long['Year'] >= 1980) & (data_long['Year'] <= 2014) &
    (data_long['Indicator'].isin(['Emissions','Temperature','GDP','Energy','Disasters']))].drop(columns="Label").copy()
wide_mex = CO2_indic_mex_facet.pivot(index='Year', columns='Indicator', values='Value')
scaled_mex = wide_mex.copy()
for col in scaled_mex.columns:
    scaled_mex[col] = pd.to_numeric(scaled_mex[col], errors='coerce')

#all available indicators except emissions
indicators = [col for col in scaled_mex.columns if col != "Emissions"]
choice = st.selectbox("Choose an indicator to compare with CO₂ Emissions:", indicators)

df_clean = scaled_mex[['Emissions', choice]].dropna()
df_clean['Emissions_scaled'] = (df_clean['Emissions'] - df_clean['Emissions'].mean()) / df_clean['Emissions'].std()
df_clean['Indicator_scaled'] = (df_clean[choice] - df_clean[choice].mean()) / df_clean[choice].std()

st.write("Emissions_scaled:")
st.write(df_clean['Emissions_scaled'].tolist())

st.write(f"{choice} scaled:")
st.write(df_clean['Indicator_scaled'].tolist())

st.write("Dtypes:")
st.write(df_clean.dtypes)

r = np.corrcoef(df_clean['Emissions_scaled'], df_clean['Indicator_scaled'])[0,1]
st.write(f"Correlation coefficient between Emissions and {choice}: **{r:.2f}**")

fig_corr, ax = plt.subplots(figsize=(8, 6))
sns.regplot(df_clean, x="Emissions_scaled", y="Indicator_scaled",
        scatter_kws={"color": "black", "s": 15},
        line_kws={"color": "blue", "linewidth": 2}, ci=None, ax=ax)
ax.set_title(f"Mexico CO₂ Emissions vs {choice} (Scaled)", fontsize=16)
ax.set_xlabel("Scaled Emissions (Metric Tonnes)", fontsize=14)
ax.set_ylabel(f"Scaled {choice}", fontsize=14)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
st.pyplot(fig_corr)



