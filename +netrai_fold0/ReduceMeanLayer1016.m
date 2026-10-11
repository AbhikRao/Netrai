classdef ReduceMeanLayer1016 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1016';
        end
    end


    methods
        function this = ReduceMeanLayer1016(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_238'};
        end

        function [x_model_backbone_238] = predict(this, x_model_backbone_232)
            if isdlarray(x_model_backbone_232)
                x_model_backbone_232 = stripdims(x_model_backbone_232);
            end
            x_model_backbone_232NumDims = 4;
            x_model_backbone_232 = netrai_fold0.ops.permuteInputVar(x_model_backbone_232, [4 3 1 2], 4);

            [x_model_backbone_238, x_model_backbone_238NumDims] = ReduceMeanGraph1048(this, x_model_backbone_232, x_model_backbone_232NumDims, false);
            x_model_backbone_238 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_238, [3 4 2 1], 4);

            x_model_backbone_238 = dlarray(single(x_model_backbone_238), 'SSCB');
        end

        function [x_model_backbone_238] = forward(this, x_model_backbone_232)
            if isdlarray(x_model_backbone_232)
                x_model_backbone_232 = stripdims(x_model_backbone_232);
            end
            x_model_backbone_232NumDims = 4;
            x_model_backbone_232 = netrai_fold0.ops.permuteInputVar(x_model_backbone_232, [4 3 1 2], 4);

            [x_model_backbone_238, x_model_backbone_238NumDims] = ReduceMeanGraph1048(this, x_model_backbone_232, x_model_backbone_232NumDims, true);
            x_model_backbone_238 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_238, [3 4 2 1], 4);

            x_model_backbone_238 = dlarray(single(x_model_backbone_238), 'SSCB');
        end

        function [x_model_backbone_238, x_model_backbone_238NumDims1050] = ReduceMeanGraph1048(this, x_model_backbone_232, x_model_backbone_232NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1049, x_model_backbone_232NumDims);
            xMean = mean(x_model_backbone_232, dims);
            x_model_backbone_238 = xMean;
            x_model_backbone_238NumDims = x_model_backbone_232NumDims;

            % Set graph output arguments
            x_model_backbone_238NumDims1050 = x_model_backbone_238NumDims;

        end

    end

end