"""
This is the file that contains the different gap-filling techniques presented in Amber's paper.
See: https://github.com/amberJ99/Gapfilling_debiasingERA5/tree/main

This file contains functions to perform the gap-filling of a single gap. Besides some help-functions, each gap-filling technique has its own function.

!!!!! GF TECHNIQUES ONLY WORK PROPERLY IF THE ONLY NAN VALUES ARE THE MISSING VALUES OF A SINGLE GAP WHICH IS FILLED !!!!!
!!!!! GF TECHNIQUES ASSUME HOURLY DATA !!!!!

"""

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def determine_LP(begingap, endgap, seasonal_span, positioning):
    """
    Determines the datetime values of the learning period (LP).

    Parameters
    ----------
    begingap : datetime
        First datetimestamp of the gap.
    endgap : datetime
        Last datetimestamp of the gap.
    seasonal_span : integer
        Size of the learning period expressed in number of days.
        This amount of days will be taken before or after the gap, or a combination of both.
    positioning : string
        Positioning of the learning period. Possibilities:
            -'left': the LP is placed before the gap
            -'right': the LP is placed after the gap
            -'both': the LP is placed symmetrical around the gap
            -'separate': the LP is divided in two parts, one part before the gap and one part after the gap (each with a size seasonal_span/2)

    Returns
    -------
    LP : list of date ranges
        The datetimes of the learning period.
        For left, right and both this list wil only contain one datetime range.
        For separate this list contains of two datetime ranges.

    """

    # Determine begin and end learning period
    if positioning in ("left", "right"):
        beginLP = begingap - dt.timedelta(days=seasonal_span)
        endLP = endgap + dt.timedelta(days=seasonal_span)
    elif positioning in ("both", "separate"):
        beginLP = begingap - dt.timedelta(days=seasonal_span / 2)
        endLP = endgap + dt.timedelta(days=seasonal_span / 2)
    else:
        print("ERROR: positioning must be left, right, both or separate.")

    # Determine datetimes of learning period
    datesLP_L = pd.date_range(
        start=beginLP, end=begingap, freq="h", name="DateTime", inclusive="left"
    )
    datesLP_R = pd.date_range(
        start=endgap, end=endLP, freq="h", name="DateTime", inclusive="right"
    )

    # Make list of (parts of) learning period
    if positioning == "left":
        LP = [datesLP_L]
    elif positioning == "right":
        LP = [datesLP_R]
    elif positioning == "both":
        LP = [datesLP_L.union(datesLP_R)]
    elif positioning == "separate":
        LP = [datesLP_L, datesLP_R]

    return LP


def GF_linint(df, name_gapped):
    """
    Fills the missing values (NaN) by linear interpolation.

    Example:
        GF_linint(Turku, 'Betel_gap')

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with DateTime as index and at least one column with the gapped data.
    name_gapped : string
        Name of the column with the gapped data.

    Returns
    -------
    df : pandas dataframe
        The original dataframe, but with an extra column names 'P_linint' with the gap-filled data.

    """

    df = df.copy()  # Take a copy or original df in main will also be altered
    df.loc[:, "P_linint"] = df[name_gapped].interpolate()
    return df


def GF_fillmodel(df, name_gapped, name_model):
    """
    Fills the missing values (NaN) by filling in the corresponding undebiased model values.

    Example:
        GF_fillmodel(Turku, 'Betel_gap', 'Betel_ERA5')

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with DateTime as index and at least one column with the gapped data and one column with the model data.
    name_gapped : string
        Name of the column with the gapped data.
    name_model : string
        Name of the column with the model data.

    Returns
    -------
    df : pandas dataframe
        The original dataframe, but with an extra column names 'P_fillmodel' with the gap-filled data.

    """

    df = df.copy()  # Take a copy or original df in main will also be altered
    df.loc[:, "P_fillmodel"] = df[name_gapped].fillna(value=df[name_model])
    return df


def GF_debmodelReg(
    df,
    name_gapped,
    name_model,
    seasonal_span=30,
    time_variation=0,
    positioning="both",
    degree=1,
    plot=False,
):
    """
    Performs gap-filling by debiasing model data by calculating a regression between the observations and model data of a certain learning period.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with DateTime as index and at least one column with the gapped data and one column with model data.
    name_gapped : string
        Name of the column with the gapped data.
    name_model : string
        Name of the column with the model data.
    seasonal_span : integer, optional
        Size of the learning period expressed in number of days. This amount of days will be taken before or after the gap, or a combination of both (depending on the positioning).
        The default is 30.
    time_variation : integer, optional
        Variation in hour when calculating the regression for a certain hour X.
        Example:
        For the regression of hour X, the values which belong to a datetime with hour within the time window [X-time_variation, X+time_variation] will be selected and used.
        If time_variation > 11, there won't be a distinction between the different hours of the gap (only one regression is calculated with the values of all the datetimes in the learning period determined by seasonal_span).
        The default is 0.
    positioning : string
        Positioning of the learning period. Possibilities:
            -'left': the LP is placed before the gap
            -'right': the LP is placed after the gap
            -'both': the LP is placed symmetrical around the gap
            -'separate': the LP is divided in two parts, one part before the gap and one part after the gap (each with a size seasonal_span/2)
        The default is 'both'.
    degree : integer, optional
        Degree of the regression.
        The default is 1.
    plot : boolean, optional
        If plot==True, a plot is made with the observations and model data within the time window, including the calculated regression.
        The default is False.

    Returns
    -------
    df : pandas dataframe
        The original dataframe, but with an extra column names 'P_debmodelReg' with the gap-filled data.

    """

    df = df.copy()  # Take a copy or original df in main will also be altered

    # 1. Determine begin and end of gap
    datesgap = df.index[pd.isnull(df[name_gapped])]
    hoursgap = datesgap.hour.unique()
    begingap = datesgap[0]
    endgap = datesgap[-1]

    # 2. Determine LP
    LP = determine_LP(begingap, endgap, seasonal_span, positioning)

    # 3. Debias ERA5

    # Initialise the dataframe to store debiased values
    debias = pd.DataFrame(index=datesgap)
    debias.loc[:, "ERA5"] = df.loc[datesgap, name_model]

    # Debias for each part of the learning period separately
    for i, datesLP in enumerate(LP):
        # Initialise dataframe in which coefficients of the regression will be stored
        dfcoeff = pd.DataFrame(index=hoursgap, columns=np.arange(degree, -1, -1))

        # If time_variation>11: one regression will be performed that is valid for every hour of the day
        if time_variation > 11:
            # Determine coefficients through regression
            X = df.loc[datesLP, name_model].array
            Y = df.loc[datesLP, name_gapped].array
            coeff = np.polyfit(X, Y, degree)
            dfcoeff.loc[:, :] = coeff

            # Plot the datapoints and regression line
            if plot == True:
                reg = np.polyval(coeff, X)
                plt.plot(X, Y, "o", label="data")
                plt.plot(X, reg, label="regression with degree " + str(degree))
                plt.legend()
                plt.xlabel("Model temperature (°C)")
                plt.ylabel("Observational temperature (°C)")
                plt.show()

        # If time_variation<=11: for every hour in the gap a regression will be performed
        else:
            datesLP = (
                datesLP.to_series()
            )  # To be able to slice DateTime given a condition
            for h in hoursgap:
                # Select only the datetimes with the hour in the time window
                beginW = h - time_variation
                endW = h + time_variation
                if (0 <= beginW) & (
                    endW < 24
                ):  # Time window contains no day transition
                    datesLPhour = datesLP.loc[
                        (beginW <= datesLP.dt.hour) & (datesLP.dt.hour <= endW)
                    ].index
                elif beginW < 0:  # Time window contains a day transition
                    datesLPhour = datesLP.loc[
                        (beginW % 24 <= datesLP.dt.hour) | (datesLP.dt.hour <= endW)
                    ].index
                elif 23 < endW:  # Time window contains a day transition
                    datesLPhour = datesLP.loc[
                        (beginW <= datesLP.dt.hour) | (datesLP.dt.hour <= endW % 24)
                    ].index

                # Determine coefficients through regression
                X = df.loc[datesLPhour, name_model].array
                Y = df.loc[datesLPhour, name_gapped].array
                coeff = np.polyfit(X, Y, degree)
                dfcoeff.loc[h, :] = coeff

                # Plot the datapoints and regression line
                if plot:
                    reg = np.polyval(coeff, np.sort(X))
                    plt.plot(X, Y, "o", label="data")
                    plt.plot(
                        np.sort(X), reg, label="regression with degree " + str(degree)
                    )
                    plt.legend()
                    plt.xlabel("Model temperature (°C)")
                    plt.ylabel("Observational temperature (°C)")
                    plt.show()

        # Debias ERA5-data using the coefficients
        debias.loc[:, "debmodel_" + str(i)] = debias.apply(
            lambda x: np.polyval(dfcoeff.loc[x.name.hour], x.ERA5), axis="columns"
        )

    # 4. Determine final debiased value

    if positioning in ["left", "right", "both"]:
        debias.rename(columns={"debmodel_0": "debmodel"}, inplace=True)
    if positioning == "separate":
        # Calculate weights for regression left and regression right
        if len(datesgap) == 1:
            wL = [0.5]
            wR = [0.5]
        else:
            wR = np.linspace(0, 1, num=len(datesgap), endpoint=True)
            wL = 1 - wR
        weights = pd.DataFrame(list(zip(wL, wR)), index=datesgap, columns=["wL", "wR"])
        # Multiply each column with their weights and add the two columns up
        debias.loc[:, "debmodel_0"] = debias.loc[:, "debmodel_0"] * weights.wL
        debias.loc[:, "debmodel_1"] = debias.loc[:, "debmodel_1"] * weights.wR
        debias.loc[:, "debmodel"] = (
            debias.loc[:, "debmodel_0"] + debias.loc[:, "debmodel_1"]
        )

    # 5. Fill in debiased ERA5 to fill gap

    df.loc[:, "P_debmodelReg"] = df.loc[:, name_gapped]
    df.loc[datesgap, "P_debmodelReg"] = debias.debmodel

    return df


def GF_debmodelMeanbias(
    df,
    name_gapped,
    name_model,
    seasonal_span=30,
    time_variation=0,
    positioning="both",
    plot=False,
):
    """
    Performs gap-filling bij debiasing model data by calculating a mean bias (which depends on the hour of the day) for the model data of a certain learning period.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with DateTime as index and at least one column with the gapped data and one column with model data.
    name_gapped : string
        Name of the column with the gapped data.
    name_model : string
        Name of the column with the model data.
    seasonal_span : integer, optional
        Size of the learning period expressed in number of days. This amount of days will be taken before or after the gap, or a combination of both (depending on the positioning).
        The default is 30.
    time_variation : integer, optional
        Variation in hour when calculating the mean bias for a certain hour X. In practice this is done by recalculating the mean bias in a rolling window.
        Example:
        For the mean bias of hour X, the values which belong to a datetime with hour within the time window [X-time_variation, X+time_variation] will contribute.
        If time_variation > 11, there won't be a distinction between the different hours of the gap and each hour has the same mean bias.
        The default is 0.
    positioning : string
        Positioning of the learning period. Possibilities:
            -'left': the LP is placed before the gap
            -'right': the LP is placed after the gap
            -'both': the LP is placed symmetrical around the gap
            -'separate': the LP is divided in two parts, one part before the gap and one part after the gap (each with a size seasonal_span/2)
        The default is 'both'.
    plot : boolean, optional
        If plot==True, a plot is made of the mean bias in function of the hour of the day.
        The default is False.

    Returns
    -------
    df : pandas dataframe
        The original dataframe, but with an extra column names 'P_debmodelMeanbias' with the gap-filled data.


    """

    df = df.copy()  # Take a copy or original df in main will also be altered

    # 1. Determine begin and end of gap
    datesgap = df.index[pd.isnull(df[name_gapped])]
    begingap = datesgap[0]
    endgap = datesgap[-1]

    # 2. Determine LP
    LP = determine_LP(begingap, endgap, seasonal_span, positioning)

    # 3. Debias ERA5

    # Initialise the dataframe to store debiased values
    debias = pd.DataFrame(index=datesgap)
    debias.loc[:, "ERA5"] = df.loc[datesgap, name_model]

    # Debias for each part of the learning period separately
    for i, datesLP in enumerate(LP):
        # Calculate difference for each timestamp of the learning period
        bias = pd.DataFrame(index=datesLP)
        bias.loc[:, "Bias"] = df.loc[datesLP, name_model] - df.loc[datesLP, name_gapped]
        bias.loc[:, "Hour"] = bias.index.map(lambda x: x.hour)

        # Calculate the mean bias for each hour
        biasHour = bias.groupby("Hour").mean()

        # Calculate mean bias in time window
        if time_variation > 11:
            biasHour.loc[:, "Bias"] = biasHour["Bias"].mean()
        elif time_variation != 0:
            # Determine size of time window
            rollingW = 1 + 2 * time_variation
            # Extend overview of mean bias per hour because we have periodic conditions
            biasHourpast = biasHour.copy()
            biasHourpast.set_index(biasHourpast.index.to_series() - 24, inplace=True)
            biasHourfuture = biasHour.copy()
            biasHourfuture.set_index(
                biasHourfuture.index.to_series() + 24, inplace=True
            )
            biasextend = pd.concat([biasHourpast, biasHour, biasHourfuture])
            # Determine mean using rolling window en drop the extensions
            biasHour = (
                biasextend.rolling(rollingW, min_periods=rollingW, center=True)
                .mean()
                .drop(biasHourpast.index.union(biasHourfuture.index))
            )

        # Make a plot of the mean bias per hour
        if plot == True:
            plt.plot(biasHour.index, biasHour.Bias, "o")
            plt.xlabel("Hour")
            plt.ylabel("Bias")
            plt.title("Mean bias of learning period number " + str(i))
            plt.show()

        # Debias ERA5 with the calculated mean bias
        debias.loc[:, "debmodel_" + str(i)] = debias.apply(
            lambda x: (x.ERA5 - biasHour.loc[x.name.hour, "Bias"]), axis="columns"
        )

    # 4. Determine final debiased value

    if positioning in ["left", "right", "both"]:
        debias.rename(columns={"debmodel_0": "debmodel"}, inplace=True)
    if positioning == "separate":
        # Calculate weights for left and right
        if len(datesgap) == 1:
            wL = [0.5]
            wR = [0.5]
        else:
            wR = np.linspace(0, 1, num=len(datesgap), endpoint=True)
            wL = 1 - wR
        weights = pd.DataFrame(list(zip(wL, wR)), index=datesgap, columns=["wL", "wR"])
        # Multiply each column with their weights and add the two columns up
        debias.loc[:, "debmodel_0"] = debias.loc[:, "debmodel_0"] * weights.wL
        debias.loc[:, "debmodel_1"] = debias.loc[:, "debmodel_1"] * weights.wR
        debias.loc[:, "debmodel"] = (
            debias.loc[:, "debmodel_0"] + debias.loc[:, "debmodel_1"]
        )

    # 5. Fill in debiased ERA5 to fill gap

    df.loc[:, "P_debmodelMeanbias"] = df.loc[:, name_gapped]
    df.loc[datesgap, "P_debmodelMeanbias"] = debias.debmodel

    return df


def GF_debmodelTvar(
    df, name_gapped, name_model, seasonal_span=30, time_variation=0, positioning="both"
):
    """
    Performs gap-filling by debiasing model data by calculating a weighted mean bias (which depends on the hour of the day) for the model data of a certain learning period.
    The weights are determined based on similarities in daily temperature variation between the days in the learning period and the day of the filled missing value.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with DateTime as index and at least one column with the gapped data and one column with model data.
    name_gapped : string
        Name of the column with the gapped data.
    name_model : string
        Name of the column with the model data.
    seasonal_span : integer, optional
        Size of the learning period expressed in number of days. This amount of days will be taken before or after the gap, or a combination of both (depending on the positioning).
        The default is 30.
    time_variation : integer, optional
        Variation in hour when calculating the mean bias for a certain hour X. In practice this is done by recalculating the mean bias in a rolling window.
        Example:
        For the weighted mean bias of hour X, the values which belong to a datetime with hour within the time window [X-time_variation, X+time_variation] will contribute.
        If time_variation > 11, there won't be a distinction between the different hours of the gap and each hour has the same mean bias.
        The default is 0.
    positioning : string
        Positioning of the learning period. Possibilities:
            -'left': the LP is placed before the gap
            -'right': the LP is placed after the gap
            -'both': the LP is placed symmetrical around the gap
            -'separate': the LP is divided in two parts, one part before the gap and one part after the gap (each with a size seasonal_span/2)
        The default is 'both'.

    Returns
    -------
    df : pandas dataframe
        The original dataframe, but with an extra column names 'P_debmodelTvar' with the gap-filled data.

    """

    df = df.copy()  # Take a copy or original df in main will also be altered

    # 1. Determine begin and end of gap
    datesgap = df.index[pd.isnull(df[name_gapped])]
    begingap = datesgap[0]
    endgap = datesgap[-1]

    # 2. Determine LP
    LP = determine_LP(begingap, endgap, seasonal_span, positioning)

    # 3. Debias ERA5

    # Initialise the dataframe to store debiased values
    debias = pd.DataFrame(index=datesgap)
    debias.loc[:, "ERA5"] = df.loc[datesgap, name_model]
    debias.loc[:, "Hour"] = debias.index.hour
    debias.loc[:, "Date"] = debias.index.date

    # Debias for each part of the learning period separately
    for i, datesLP in enumerate(LP):
        # Calculate difference for each timestamp of the learning period
        bias = df.loc[datesLP, [name_model, name_gapped]].copy()
        bias.loc[:, "Bias"] = bias[name_model] - bias[name_gapped]
        bias.loc[:, "Hour"] = bias.index.map(lambda x: x.hour)
        bias.loc[:, "Date"] = bias.index.date

        # Calculate Tvar for each day (LP + gap)
        alldates = datesLP.union(datesgap)
        uniquedays = np.unique(alldates.date)
        df_fulldays = df.loc[np.isin(df.index.date, uniquedays), [name_model]]
        df_fulldays.loc[:, "Date"] = df_fulldays.index.date
        df_minandmax = df_fulldays.groupby("Date").agg(["min", "max"])
        Tvar = df_minandmax[name_model]["max"] - df_minandmax[name_model]["min"]
        Tvar = Tvar.to_frame(name="Tvar")

        # Debias for each day of the gap separately (because Tvar/weights will differ for each day)
        # Initialise dataframe to store the bias per hour, for every day of the gap
        hourlybiastotal = pd.DataFrame(index=range(0, 24))
        for daygap in np.unique(datesgap.date):
            # For every new day of the gap, start again from the original dataframes
            Tvarnew = Tvar.copy()
            biasnew = bias.copy()

            # Calculate the weights for each day
            Tvarnew.loc[:, "absdiff"] = Tvarnew.Tvar.map(
                lambda x: abs(x - Tvar.loc[daygap, "Tvar"])
            )
            Tvarnew["absdiff"] = Tvarnew["absdiff"].replace(0, np.nan)
            Tvarnew.loc[:, "weight"] = Tvarnew.absdiff.map(lambda x: 1 / x)

            # Multiply bias of every datetime of LP with the given weight
            biasnew.loc[:, "weight"] = biasnew.apply(
                lambda x: Tvarnew.loc[x.Date, "weight"], axis=1
            )
            biasnew.loc[:, "wtimesb"] = biasnew.apply(
                lambda x: x.Bias * x.weight, axis=1
            )

            # Calculate sum over weighted biases and sum over all weights for every hour
            meanbiashour = biasnew[["Hour", "weight", "wtimesb"]].groupby("Hour").sum()

            # Take into account timewindow
            if time_variation > 11:
                meanbiashour.loc[:, "wtimesb"] = meanbiashour["wtimesb"].sum()
                meanbiashour.loc[:, "weight"] = meanbiashour["weight"].sum()
            elif time_variation != 0:
                # Determine size of time window
                rollingW = 1 + 2 * time_variation
                # Extend overview of sum over weighted biases per hour because we have periodic conditions
                biaspast = meanbiashour.copy()
                biaspast.set_index(biaspast.index.to_series() - 24, inplace=True)
                biasfuture = meanbiashour.copy()
                biasfuture.set_index(biasfuture.index.to_series() + 24, inplace=True)
                biasextend = pd.concat([biaspast, meanbiashour, biasfuture])
                # Determine sum using rolling window en drop the extensions
                meanbiashour = (
                    biasextend.rolling(rollingW, min_periods=rollingW, center=True)
                    .sum()
                    .drop(biaspast.index.union(biasfuture.index))
                )

            # Calculate weighted bias by dividing sum of wtimesb by the sum of the weights
            meanbiashour.loc[:, "weightedbias"] = meanbiashour.apply(
                lambda x: x.wtimesb / x.weight, axis=1
            )

            # Store weighted biases for that day of the gap in the overall dataframe
            hourlybiastotal.loc[:, daygap] = meanbiashour.weightedbias

        # Debias ERA5
        debias.loc[:, "debmodel_" + str(i)] = debias.apply(
            lambda x: (x.ERA5 - hourlybiastotal.loc[x.Hour, x.Date]), axis="columns"
        )

    # 4. Determine final debiased value

    if positioning in ["left", "right", "both"]:
        debias.rename(columns={"debmodel_0": "debmodel"}, inplace=True)
    if positioning == "separate":
        # Calculate weights for left and right
        if len(datesgap) == 1:
            wL = [0.5]
            wR = [0.5]
        else:
            wR = np.linspace(0, 1, num=len(datesgap), endpoint=True)
            wL = 1 - wR
        weights = pd.DataFrame(list(zip(wL, wR)), index=datesgap, columns=["wL", "wR"])
        # Multiply each column with their weights and add the two columns up
        debias.loc[:, "debmodel_0"] = debias.loc[:, "debmodel_0"] * weights.wL
        debias.loc[:, "debmodel_1"] = debias.loc[:, "debmodel_1"] * weights.wR
        debias.loc[:, "debmodel"] = (
            debias.loc[:, "debmodel_0"] + debias.loc[:, "debmodel_1"]
        )

    # 5. Fill in debiased ERA5 to fill gap

    df.loc[:, "P_debmodelTvar"] = df.loc[:, name_gapped]
    df.loc[datesgap, "P_debmodelTvar"] = debias.debmodel

    return df
