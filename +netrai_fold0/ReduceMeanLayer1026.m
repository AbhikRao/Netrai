classdef ReduceMeanLayer1026 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1026';
        end
    end


    methods
        function this = ReduceMeanLayer1026(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_387'};
        end

        function [x_model_backbone_387] = predict(this, x_model_backbone_381)
            if isdlarray(x_model_backbone_381)
                x_model_backbone_381 = stripdims(x_model_backbone_381);
            end
            x_model_backbone_381NumDims = 4;
            x_model_backbone_381 = netrai_fold0.ops.permuteInputVar(x_model_backbone_381, [4 3 1 2], 4);

            [x_model_backbone_387, x_model_backbone_387NumDims] = ReduceMeanGraph1078(this, x_model_backbone_381, x_model_backbone_381NumDims, false);
            x_model_backbone_387 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_387, [3 4 2 1], 4);

            x_model_backbone_387 = dlarray(single(x_model_backbone_387), 'SSCB');
        end

        function [x_model_backbone_387] = forward(this, x_model_backbone_381)
            if isdlarray(x_model_backbone_381)
                x_model_backbone_381 = stripdims(x_model_backbone_381);
            end
            x_model_backbone_381NumDims = 4;
            x_model_backbone_381 = netrai_fold0.ops.permuteInputVar(x_model_backbone_381, [4 3 1 2], 4);

            [x_model_backbone_387, x_model_backbone_387NumDims] = ReduceMeanGraph1078(this, x_model_backbone_381, x_model_backbone_381NumDims, true);
            x_model_backbone_387 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_387, [3 4 2 1], 4);

            x_model_backbone_387 = dlarray(single(x_model_backbone_387), 'SSCB');
        end

        function [x_model_backbone_387, x_model_backbone_387NumDims1080] = ReduceMeanGraph1078(this, x_model_backbone_381, x_model_backbone_381NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1079, x_model_backbone_381NumDims);
            xMean = mean(x_model_backbone_381, dims);
            x_model_backbone_387 = xMean;
            x_model_backbone_387NumDims = x_model_backbone_381NumDims;

            % Set graph output arguments
            x_model_backbone_387NumDims1080 = x_model_backbone_387NumDims;

        end

    end

end