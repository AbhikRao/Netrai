classdef ReduceMeanLayer1020 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1020';
        end
    end


    methods
        function this = ReduceMeanLayer1020(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_298'};
        end

        function [x_model_backbone_298] = predict(this, x_model_backbone_292)
            if isdlarray(x_model_backbone_292)
                x_model_backbone_292 = stripdims(x_model_backbone_292);
            end
            x_model_backbone_292NumDims = 4;
            x_model_backbone_292 = netrai_fold0.ops.permuteInputVar(x_model_backbone_292, [4 3 1 2], 4);

            [x_model_backbone_298, x_model_backbone_298NumDims] = ReduceMeanGraph1060(this, x_model_backbone_292, x_model_backbone_292NumDims, false);
            x_model_backbone_298 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_298, [3 4 2 1], 4);

            x_model_backbone_298 = dlarray(single(x_model_backbone_298), 'SSCB');
        end

        function [x_model_backbone_298] = forward(this, x_model_backbone_292)
            if isdlarray(x_model_backbone_292)
                x_model_backbone_292 = stripdims(x_model_backbone_292);
            end
            x_model_backbone_292NumDims = 4;
            x_model_backbone_292 = netrai_fold0.ops.permuteInputVar(x_model_backbone_292, [4 3 1 2], 4);

            [x_model_backbone_298, x_model_backbone_298NumDims] = ReduceMeanGraph1060(this, x_model_backbone_292, x_model_backbone_292NumDims, true);
            x_model_backbone_298 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_298, [3 4 2 1], 4);

            x_model_backbone_298 = dlarray(single(x_model_backbone_298), 'SSCB');
        end

        function [x_model_backbone_298, x_model_backbone_298NumDims1062] = ReduceMeanGraph1060(this, x_model_backbone_292, x_model_backbone_292NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1061, x_model_backbone_292NumDims);
            xMean = mean(x_model_backbone_292, dims);
            x_model_backbone_298 = xMean;
            x_model_backbone_298NumDims = x_model_backbone_292NumDims;

            % Set graph output arguments
            x_model_backbone_298NumDims1062 = x_model_backbone_298NumDims;

        end

    end

end