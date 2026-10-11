classdef ReduceMeanLayer1022 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1022';
        end
    end


    methods
        function this = ReduceMeanLayer1022(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_327'};
        end

        function [x_model_backbone_327] = predict(this, x_model_backbone_321)
            if isdlarray(x_model_backbone_321)
                x_model_backbone_321 = stripdims(x_model_backbone_321);
            end
            x_model_backbone_321NumDims = 4;
            x_model_backbone_321 = netrai_fold0.ops.permuteInputVar(x_model_backbone_321, [4 3 1 2], 4);

            [x_model_backbone_327, x_model_backbone_327NumDims] = ReduceMeanGraph1066(this, x_model_backbone_321, x_model_backbone_321NumDims, false);
            x_model_backbone_327 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_327, [3 4 2 1], 4);

            x_model_backbone_327 = dlarray(single(x_model_backbone_327), 'SSCB');
        end

        function [x_model_backbone_327] = forward(this, x_model_backbone_321)
            if isdlarray(x_model_backbone_321)
                x_model_backbone_321 = stripdims(x_model_backbone_321);
            end
            x_model_backbone_321NumDims = 4;
            x_model_backbone_321 = netrai_fold0.ops.permuteInputVar(x_model_backbone_321, [4 3 1 2], 4);

            [x_model_backbone_327, x_model_backbone_327NumDims] = ReduceMeanGraph1066(this, x_model_backbone_321, x_model_backbone_321NumDims, true);
            x_model_backbone_327 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_327, [3 4 2 1], 4);

            x_model_backbone_327 = dlarray(single(x_model_backbone_327), 'SSCB');
        end

        function [x_model_backbone_327, x_model_backbone_327NumDims1068] = ReduceMeanGraph1066(this, x_model_backbone_321, x_model_backbone_321NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1067, x_model_backbone_321NumDims);
            xMean = mean(x_model_backbone_321, dims);
            x_model_backbone_327 = xMean;
            x_model_backbone_327NumDims = x_model_backbone_321NumDims;

            % Set graph output arguments
            x_model_backbone_327NumDims1068 = x_model_backbone_327NumDims;

        end

    end

end