"""

This file contains the functions to perform the evaluation of the gap-filling techniques

"""

from amber_gap_filling.gf_techniques import (
    GF_linint,
    GF_fillmodel,
    GF_debmodelReg,
    GF_debmodelMeanbias,
    GF_debmodelTvar,
)

import datetime as dt
import pandas as pd
import numpy as np
import random as r
import matplotlib.pyplot as plt


def makegap(df, name_fulldata, lengthhour, starttime):
    """
    Makes a gap in data by replacing known values by NaN.

    Example:
        makegap(Turku, 'Betel', 5, '2020-02-05 15:00:00')

    Parameters
    ----------
    df : pandas dataframe
        Dataframe has DateTime as index, with at least one column with the values of the complete timeseries.
    name_fulldata : string
        The name of the column that consists of the complete data in which a gap will be made.
    lengthhour : integer
        Amount of hours that will be missing.
        (Because DateTime is given in steps of one hour, this will be the amount of values that will be missing).
    starttime : string
        Startdate and -time of the gap, given in format 'YYYY-MM-DD HH:MM:SS'.

    Returns
    -------
    df : pandas dataframe
        Original dataframe, but with an extra column named name_fulldata + '_gap' which consists of the data with gap.

    """

    DateB = dt.datetime.strptime(starttime, "%Y-%m-%d %H:%M:%S")
    DateE = DateB + dt.timedelta(hours=lengthhour)
    df = df.copy()  # Take a copy or original df in main will also be altered
    df.loc[:, name_fulldata + "_gap"] = df.loc[:, name_fulldata]
    df.loc[(df.index >= DateB) & (df.index < DateE), name_fulldata + "_gap"] = float(
        "nan"
    )

    return df


def calculate_error(df, name_fulldata, name_gapped, listerror, listtechnique):
    """
    Calculates the error of the gap-filling of a single gap for each applied technique, given the original data, the gapped data and the gap-filled data.

    Example:
        calculate_error(df_result, 'Betel', 'Betel_gap', ['MBE', 'MSE'], ['fillmodel', 'linint'])

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with the data.
        The dataframe should at least have one column with original complete data.
        The dataframe should at least have one column with the data containing the gap.
        The dataframe should at least have for each listed technique a column named 'P_' + technique.
    name_fulldata : string
        Name of the column with the original and complete data.
    name_gapped : string
        Name of the column with the gapped data.
    listerror : string or list of strings
        Error(s) which are calculated.
        Possible errors: 'MBE', 'RMSE', 'MSE', 'MAE'.
    listtechnique : string or list of strings
        Technique(s) for which the error is calculated.
        Possible techniques: 'fillmodel', 'linint', 'debmodelReg', 'debmodelMeanbias', 'debmodelTvar'.

    Returns
    -------
    dferror: pandas dataframe.
        Dataframe with the error of the gap-filling for each technique.
        The different errors are the index, and different gap-filling techniques are the columns.

    """

    if isinstance(listerror, str):
        listerror = [listerror]
    if isinstance(listtechnique, str):
        listtechnique = [listtechnique]
    dferror = pd.DataFrame(columns=listtechnique, index=listerror)
    for error in listerror:
        for technique in listtechnique:
            difference = pd.DataFrame(
                df.loc[:, "P_" + technique] - df[name_fulldata]
            )  # Difference between predicted and actual value
            n = df[name_gapped].isnull().sum()  # Amount of timesteps of the gap
            if error == "MBE":
                ervalue = difference.iloc[:, 0].sum() / n
            if error == "RMSE":
                difference.loc[:, "square"] = difference.iloc[:, 0].map(lambda x: x**2)
                ervalue = np.sqrt(difference.loc[:, "square"].sum() / n)
            if error == "MSE":
                difference.loc[:, "square"] = difference.iloc[:, 0].map(lambda x: x**2)
                ervalue = difference.loc[:, "square"].sum() / n
            if error == "MAE":
                difference.loc[:, "absolute"] = difference.iloc[:, 0].map(
                    lambda x: abs(x)
                )
                ervalue = difference.loc[:, "absolute"].sum() / n
            dferror.loc[error, technique] = ervalue

    return dferror


def Perform_techniques(
    df,
    name_fulldata,
    name_model,
    gap_datetime,
    gap_length,
    dictionarytechniques,
    listerrors,
    plot=False,
    check=False,
):
    """
    Makes a gap in the data, performs multiple gap filling techniques on the gapped data and calculates the errors.

    Example:
        Perform_techniques(Turku, 'Betel', 'Betel_ERA5', '2020-02-05 15:00:00', 8, dictionary, ['MBE', 'MSE'], True, True)

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with DateTime as index and at least one column with original complete data and one column with model data.
    name_fulldata : string
        Name of the column with the original and complete data.
    name_model : string
        Name of the column with the model data. Give False if this column is not needed and not present (for example for linint).
    gap_datetime : string
        Begin date of the gap, given in format 'YYYY-mm-dd HH:MM:SS'.
    gap_length : integer
        Length of the gap in hours (!!! assuming steps of data is one hour !!!).
    dictionarytechniques : dictionary
        Dictionary with the names of the techniques as keys. Possibilities: 'linint', 'fillmodel', 'debmodelReg', 'debmodelMeanbias', 'debmodelTvar'.
        If for a certain key the dictionary consists of a list, the values of the list are used as arguments of the function (extended with the dataframe and the name of the columns as argument).
        If it is not a list, only the dataframe will be an argument for the function.
    listerrors : list of strings
        List of errors which are calculated. Possibilities: 'MBE', 'RMSE', 'MSE', 'MAE'.
    plot : boolean, optional
        If plot==True: the original complete data + ERA5 data + filled data is plotted for the gap.
        The default is False.
    check : boolean, optional
        If check == True everytime a gap filling technique is performed this is printed on the screen.
        The default is False.

    Returns
    -------
    dfgap : pandas dataframe.
        Original dataframe df, but with extra columns.
        One extra column named name_fulldata + '_gap' with the data with gap.
        One extra column for each gap-filling technique performed, named 'P_' + name_technique.
    dferror: pandas dataframe.
        Dataframe with the error of the gap-filling for each technique.
        The different errors are the index, and different gap-filling techniques are the columns.
    """

    df = df.copy()  # Take a copy or original df in main will also be altered

    # Make a gap
    dfgap = makegap(df, name_fulldata, gap_length, gap_datetime)

    # Perform gap-filling techniques
    listtechniques = []
    for technique in dictionarytechniques:
        listtechniques.append(technique)  # Make a list of the names of the techniques

        if technique == "linint":
            dfgap = GF_linint(dfgap, name_fulldata + "_gap")
        elif technique == "fillmodel":
            dfgap = GF_fillmodel(dfgap, name_fulldata + "_gap", name_model)
        elif technique == "debmodelReg":
            arg = [dfgap, name_fulldata + "_gap", name_model]
            arg_extra = dictionarytechniques[technique].copy()
            arg.extend(arg_extra)
            dfgap = GF_debmodelReg(*arg)
        elif technique == "debmodelMeanbias":
            arg = [dfgap, name_fulldata + "_gap", name_model]
            arg_extra = dictionarytechniques[technique].copy()
            arg.extend(arg_extra)
            dfgap = GF_debmodelMeanbias(*arg)
        elif technique == "debmodelTvar":
            arg = [dfgap, name_fulldata + "_gap", name_model]
            arg_extra = dictionarytechniques[technique].copy()
            arg.extend(arg_extra)
            dfgap = GF_debmodelTvar(*arg)

        if check == True:
            print("The technique " + technique + " has been performed.")

    # Plot filled gap
    if plot == True:
        begindate = gap_datetime[0:10]  # Select only the date
        enddate = (
            dt.datetime.strptime(gap_datetime, "%Y-%m-%d %H:%M:%S")
            + dt.timedelta(days=1, hours=gap_length)
        ).strftime("%Y-%m-%d")
        dfgapsl = dfgap.loc[
            (dfgap.index >= dt.datetime.strptime(begindate, "%Y-%m-%d"))
            & (dfgap.index < dt.datetime.strptime(enddate, "%Y-%m-%d"))
        ]

        plottechniques = ["P_" + i for i in listtechniques]
        plottechniques.insert(0, name_fulldata)
        plottechniques.insert(1, name_model)
        plotstyle = ["--" for i in listtechniques]
        plotstyle.insert(0, "-")
        plotstyle.insert(1, "-")

        dfgapsl.loc[:, plottechniques].plot(style=plotstyle)
        plt.ylabel("Temperature (°C)")
        plt.xlabel("DateTime")
        plt.show()

    # Calculate errors
    dferror = calculate_error(
        dfgap, name_fulldata, name_fulldata + "_gap", listerrors, listtechniques
    )

    return dfgap, dferror


def Test_techniques_differentgapdates(
    df,
    name_fulldata,
    name_model,
    gap_length,
    dictionarytechniques,
    par_slicedates,
    listerrors,
    repetitions,
    check=1,
):
    """
    Calculates for a given gaplength for each GF-technique the mean error and standard error.

    Example:
    results = Test_techniques_differentgapdates(df=dataset,
                                                name_fulldata='Betel',
                                                name_model='Betel_ERA5',
                                                gap_length=5,
                                                dictionarytechniques=dictionary,
                                                par_slicedates=30,
                                                error=['MSE', 'MBE'],
                                                range_gaplengths=(1, 3, 5, 7, 10, 20, 30, 336),
                                                repetitions=1000,
                                                check=250)

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with DateTime as index, and consists of one column with the observations (must be complete!) and one column with the model data (must be complete!).
    name_fulldata : string
        Name of the column with the observations.
    name_model : string
        Name of the column with the model data. Give False if this column is not needed and not present (for example for linint).
    gap_length : integer
        Length of the gap for which the gap filling techniques are performed and evaluated.
    dictionarytechniques : dictionary
        Dictionary with the name of the different GF-techniques as key.
        Possibilities: 'linint', 'fillmodel', 'debmodelReg', 'debmodelMeanbias', 'debmodelTvar'.
        If for a certain key the dictionary consists of a list, the values of the list are used as arguments of the function (extended with the dataframe and the name of the columns as argument).
        If it is not a list, only the dataframe will be an argument for the function.
        Example: dictionarytechniques={"linint": np.nan , "fillmodel": np.nan, "debmodelReg":[12, 0, 'separate'], "debmodelMeanbias": [15, 1, 'both', True]}.
    par_slicedates : integer
        The amount of days which will be sliced from the beginning and end of the dataset, so the gap will be placed far enough from the edges of the dataset according to the seasonal span.
    listerrors : list of strings
        A list with the errors that are calculated. Possibilities: 'MBE', 'RMSE', 'MSE', 'MAE'.
    repetitions : integer
        Amount of gaps for each gap_length that are tested before calculating the mean error.
    check : integer, optional
        After this amount of gaps made a text is printed on the screen with the number of repetition we are currently at.
        The default is 1.


    Returns
    -------
    df_errors: list of pandas dataframes
        List that contains for each calculated error a pandas dataframe with for each gap-filling technique the mean error, standard error of this mean error and number of values used for this mean error.
        The dataframe has as index ['mean', 'sterr', 'len'] and the gap-filling techniques as columns.

    """

    df = df.copy()  # Take a copy or original df in main will also be altered

    # Determine possible datetimes of the beginning of the gap (so learning period can always be placed around/before/after gap)
    alldates = df.index
    possible_dates = alldates[
        (par_slicedates * 24) : -(par_slicedates * 24 + gap_length)
    ]

    # Initialise dataframes for storage of errors
    listtechniques = [technique for technique in dictionarytechniques]
    dataframe_dict = pd.DataFrame(columns=listtechniques)
    dict_errors = {error: dataframe_dict for error in listerrors}

    # Make different gaps and perform gap filling techniques
    for i in range(repetitions):
        # Take random datetime
        rand = r.randint(0, len(possible_dates) - 1)
        gap_datetime = possible_dates[rand]
        gap_datetime_str = gap_datetime.strftime("%Y-%m-%d %H:%M:%S")
        if check == 1:
            print(
                "Gap filling performed for datetime "
                + gap_datetime_str
                + " with gaplength "
                + str(gap_length)
                + " hours."
            )
        elif (i + 1) % check == 0:
            print("Repetition number = " + str(i + 1) + ".")
        # Perform gap filling techniques
        df_errors = Perform_techniques(
            df,
            name_fulldata,
            name_model,
            gap_datetime_str,
            gap_length,
            dictionarytechniques,
            listerrors,
            plot=False,
            check=False,
        )[1]
        # Store errors
        for error in listerrors:
            dict_errors[error].loc[i, :] = df_errors.loc[error, :]

    # Calculate mean error and standard deviation
    df_errors = list()
    for error in listerrors:
        dftotalerror = dict_errors[error].agg(["mean", "std", len])
        dftotalerror.loc["std", :] = dftotalerror.loc["std", :] / (
            dftotalerror.loc["len", :].map(lambda x: np.sqrt(x))
        )
        dftotalerror.rename(index={"std": "sterr"}, inplace=True)
        df_errors.append(dftotalerror)

    return df_errors


def Test_techniques_differentgaplengths(
    df,
    name_fulldata,
    name_model,
    dictionarytechniques,
    par_slicedates,
    error,
    range_gaplengths,
    repetitions,
    check=1,
    plot=False,
):
    """
    Calculates for each gaplength and for each GF-technique a mean error and standard error.

    Example:
    df_errors, df_sterr = Test_techniques_differentgaplengths(df=dataset,
                                                            name_fulldata='Betel',
                                                            name_model='Betel_ERA5',
                                                            dictionarytechniques=dictionary,
                                                            par_slicedates=30,
                                                            error='MSE',
                                                            range_gaplengths=(1, 3, 5, 7, 10, 20, 30, 336),
                                                            repetitions=1000,
                                                            check=250,
                                                            plot=False)

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with DateTime as index, and consists of one column with the observations (must be complete!) and one column with the model data (must be complete!).
    name_fulldata : string
        Name of the column with the observations.
    name_model : string
        Name of the column with the model data. Give False if this column is not needed and not present (for example for linint).
    dictionarytechniques : dictionary
        Dictionary with the name of the different GF-techniques as key.
        Possibilities: 'linint', 'fillmodel', 'debmodelReg', 'debmodelMeanbias', 'debmodelTvar'.
        If for a certain key the dictionary consists of a list, the values of the list are used as arguments of the function (extended with the dataframe and the name of the columns as argument).
        If it is not a list, only the dataframe will be an argument for the function.
        Example: dictionarytechniques={"linint": np.nan , "fillmodel": np.nan, "debmodelReg":[12, 0, 'separate'], "debmodelMeanbias": [15, 1, 'both', True]}.
    par_slicedates : integer
        The amount of days which will be sliced from the beginning and end of the dataset, so the gap will be placed far enough from the edges of the dataset according to the seasonal span.
    error : string
        The error that is calculated. Possibilities: 'MBE', 'RMSE', 'MSE', 'MAE'.
    range_gaplengths : list
        The values of the tested gaplengths.
    repetitions : integer
        Amount of gaps for each gap_length that are tested before calculating the mean error.
    check : integer, optional
        After this amount of gaps made, a text is printed on the screen with the number of repetition we are currently at.
        The default is 1.
    plot : boolean, optional
        If plot==True: a simple plot is made for the mean error in function of the gap_length for the different techniques. The default is False.

    Returns
    -------
    df_errors : pandas dataframe
        Dataframe with gaplengths as index and GF-techniques as columns, with the mean error for each combination.
    df_sterr : pandas dataframe
        Dataframe with gaplengths as index and GF-techniques as columns, with the standard error for each combination.

    """

    # Initiliase dataframe to store errors and sterr's
    listtechniques = [technique for technique in dictionarytechniques]
    df_errors = pd.DataFrame(index=range_gaplengths, columns=listtechniques)
    df_sterr = pd.DataFrame(index=range_gaplengths, columns=listtechniques)

    # Loop over all the gaplengths
    for length in range_gaplengths:
        print("Making gaps with length " + str(length) + ".")
        # Perform Test_techniques_differentgapdates
        results = Test_techniques_differentgapdates(
            df,
            name_fulldata,
            name_model,
            length,
            dictionarytechniques,
            par_slicedates,
            [error],
            repetitions,
            check,
        )
        # Store error and sterr
        error_certainlength = results[0]
        df_errors.loc[length, :] = error_certainlength.loc["mean", :]
        df_sterr.loc[length, :] = error_certainlength.loc["sterr", :]

    # If plot: plot error ifo gap_length for each technique
    if plot:
        df_errors.plot(marker="o", yerr=df_sterr, ecolor="red")
        plt.xlabel("Gaplength (hour)")
        plt.ylabel(error)
        if error == "MBE":
            plt.axhline(color="k")
        plt.show()

    return df_errors, df_sterr
