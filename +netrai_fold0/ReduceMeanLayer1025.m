classdef ReduceMeanLayer1025 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
    end

    properties (State)
    end

    properties
        Vars
        NumDims
    end


    methods(Static, Hidden)
        % Specify the path to the class that will be used for codegen
        function name = matlabCodegenRedirect(~)
            name = 'netrai_fold0.coder.ReduceMeanLayer1025';
        end
    end


    methods
        function this = ReduceMeanLayer1025(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_372'};
        end

        function [x_model_backbone_372] = predict(this, x_model_backbone_366)
            if isdlarray(x_model_backbone_366)
                x_model_backbone_366 = stripdims(x_model_backbone_366);
            end
            x_model_backbone_366NumDims = 4;
            x_model_backbone_366 = netrai_fold0.ops.permuteInputVar(x_model_backbone_366, [4 3 1 2], 4);

            [x_model_backbone_372, x_model_backbone_372NumDims] = ReduceMeanGraph1075(this, x_model_backbone_366, x_model_backbone_366NumDims, false);
            x_model_backbone_372 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_372, [3 4 2 1], 4);

            x_model_backbone_372 = dlarray(single(x_model_backbone_372), 'SSCB');
        end

        function [x_model_backbone_372] = forward(this, x_model_backbone_366)
            if isdlarray(x_model_backbone_366)
                x_model_backbone_366 = stripdims(x_model_backbone_366);
            end
            x_model_backbone_366NumDims = 4;
            x_model_backbone_366 = netrai_fold0.ops.permuteInputVar(x_model_backbone_366, [4 3 1 2], 4);

            [x_model_backbone_372, x_model_backbone_372NumDims] = ReduceMeanGraph1075(this, x_model_backbone_366, x_model_backbone_366NumDims, true);
            x_model_backbone_372 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_372, [3 4 2 1], 4);

            x_model_backbone_372 = dlarray(single(x_model_backbone_372), 'SSCB');
        end

        function [x_model_backbone_372, x_model_backbone_372NumDims1077] = ReduceMeanGraph1075(this, x_model_backbone_366, x_model_backbone_366NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1076, x_model_backbone_366NumDims);
            xMean = mean(x_model_backbone_366, dims);
            x_model_backbone_372 = xMean;
            x_model_backbone_372NumDims = x_model_backbone_366NumDims;

            % Set graph output arguments
            x_model_backbone_372NumDims1077 = x_model_backbone_372NumDims;

        end

    end

end