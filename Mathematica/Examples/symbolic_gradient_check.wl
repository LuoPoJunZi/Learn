(* Compare symbolic network gradients with central finite differences. *)

Module[
    {
        input, inputWeight, hiddenBias, outputWeight, outputBias,
        parameters, network, trainingData, loss, gradient,
        initialParameters, initialRules, lossValue, symbolicGradient,
        epsilon, numericalGradient, maximumError, learningRate,
        updatedParameters, updatedRules, initialLoss, updatedLoss,
        report, predictionPlot
    },

    parameters = {inputWeight, hiddenBias, outputWeight, outputBias};
    network = outputWeight Tanh[inputWeight input + hiddenBias] + outputBias;
    trainingData = Table[
        {value, 1/4 + (3/4) Tanh[2 value]},
        {value, {-1, -1/2, 0, 1/2, 1}}
    ];
    loss = Mean[
        ((network /. input -> #[[1]]) - #[[2]])^2 & /@ trainingData
    ];
    gradient = D[loss, #] & /@ parameters;

    initialParameters = N[{4/5, 1/10, 3/5, -1/5}];
    initialRules = Thread[parameters -> initialParameters];
    lossValue[values_List] := N[loss /. Thread[parameters -> values]];
    symbolicGradient = N[gradient /. initialRules];
    epsilon = 10.^-6;
    numericalGradient = Table[
        With[{direction = UnitVector[Length[parameters], index]},
            (lossValue[initialParameters + epsilon direction] -
                lossValue[initialParameters - epsilon direction]) / (2 epsilon)
        ],
        {index, Length[parameters]}
    ];
    maximumError = Max[Abs[symbolicGradient - numericalGradient]];

    learningRate = 0.1;
    updatedParameters = initialParameters - learningRate symbolicGradient;
    updatedRules = Thread[parameters -> updatedParameters];
    initialLoss = lossValue[initialParameters];
    updatedLoss = lossValue[updatedParameters];
    report = TestReport[{
        VerificationTest[maximumError < 10^-7, True,
            TestID -> "symbolic-vs-finite-difference"],
        VerificationTest[updatedLoss < initialLoss, True,
            TestID -> "one-step-loss-decrease"]
    }];

    Print["Parameter order: {w1, b1, w2, b2}"];
    Print["Symbolic gradient: ", symbolicGradient];
    Print["Numerical gradient: ", numericalGradient];
    Print["Maximum gradient error: ", maximumError];
    Print["Loss before update: ", initialLoss];
    Print["Loss after update:  ", updatedLoss];
    Print[report];
    If[!TrueQ[maximumError < 10^-7 && updatedLoss < initialLoss], Abort[]];

    predictionPlot = Legended[
        Show[
            Plot[
                Evaluate[{network /. initialRules, network /. updatedRules}],
                {input, -1, 1},
                PlotStyle -> {Blue, Darker[Green]},
                AxesLabel -> {"input", "prediction"},
                PlotLabel -> "A checked gradient step",
                PlotRange -> All,
                ImageSize -> Large
            ],
            ListPlot[trainingData, PlotStyle -> Black]
        ],
        Placed[
            LineLegend[{Blue, Darker[Green]}, {"Before one step", "After one step"}],
            Below
        ]
    ];
    predictionPlot
]
