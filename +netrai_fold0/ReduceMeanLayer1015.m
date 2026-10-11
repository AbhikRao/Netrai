classdef ReduceMeanLayer1015 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1015';
        end
    end


    methods
        function this = ReduceMeanLayer1015(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_224'};
        end

        function [x_model_backbone_224] = predict(this, x_model_backbone_218)
            if isdlarray(x_model_backbone_218)
                x_model_backbone_218 = stripdims(x_model_backbone_218);
            end
            x_model_backbone_218NumDims = 4;
            x_model_backbone_218 = netrai_fold0.ops.permuteInputVar(x_model_backbone_218, [4 3 1 2], 4);

            [x_model_backbone_224, x_model_backbone_224NumDims] = ReduceMeanGraph1045(this, x_model_backbone_218, x_model_backbone_218NumDims, false);
            x_model_backbone_224 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_224, [3 4 2 1], 4);

            x_model_backbone_224 = dlarray(single(x_model_backbone_224), 'SSCB');
        end

        function [x_model_backbone_224] = forward(this, x_model_backbone_218)
            if isdlarray(x_model_backbone_218)
                x_model_backbone_218 = stripdims(x_model_backbone_218);
            end
            x_model_backbone_218NumDims = 4;
            x_model_backbone_218 = netrai_fold0.ops.permuteInputVar(x_model_backbone_218, [4 3 1 2], 4);

            [x_model_backbone_224, x_model_backbone_224NumDims] = ReduceMeanGraph1045(this, x_model_backbone_218, x_model_backbone_218NumDims, true);
            x_model_backbone_224 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_224, [3 4 2 1], 4);

            x_model_backbone_224 = dlarray(single(x_model_backbone_224), 'SSCB');
        end

        function [x_model_backbone_224, x_model_backbone_224NumDims1047] = ReduceMeanGraph1045(this, x_model_backbone_218, x_model_backbone_218NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1046, x_model_backbone_218NumDims);
            xMean = mean(x_model_backbone_218, dims);
            x_model_backbone_224 = xMean;
            x_model_backbone_224NumDims = x_model_backbone_218NumDims;

            % Set graph output arguments
            x_model_backbone_224NumDims1047 = x_model_backbone_224NumDims;

        end

    end

end