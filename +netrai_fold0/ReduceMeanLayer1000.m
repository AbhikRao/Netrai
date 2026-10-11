classdef ReduceMeanLayer1000 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1000';
        end
    end


    methods
        function this = ReduceMeanLayer1000(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_b_5'};
        end

        function [x_model_backbone_b_5] = predict(this, x_model_backbone_blo)
            if isdlarray(x_model_backbone_blo)
                x_model_backbone_blo = stripdims(x_model_backbone_blo);
            end
            x_model_backbone_bloNumDims = 4;
            x_model_backbone_blo = netrai_fold0.ops.permuteInputVar(x_model_backbone_blo, [4 3 1 2], 4);

            [x_model_backbone_b_5, x_model_backbone_b_5NumDims] = ReduceMeanGraph1000(this, x_model_backbone_blo, x_model_backbone_bloNumDims, false);
            x_model_backbone_b_5 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_b_5, [3 4 2 1], 4);

            x_model_backbone_b_5 = dlarray(single(x_model_backbone_b_5), 'SSCB');
        end

        function [x_model_backbone_b_5] = forward(this, x_model_backbone_blo)
            if isdlarray(x_model_backbone_blo)
                x_model_backbone_blo = stripdims(x_model_backbone_blo);
            end
            x_model_backbone_bloNumDims = 4;
            x_model_backbone_blo = netrai_fold0.ops.permuteInputVar(x_model_backbone_blo, [4 3 1 2], 4);

            [x_model_backbone_b_5, x_model_backbone_b_5NumDims] = ReduceMeanGraph1000(this, x_model_backbone_blo, x_model_backbone_bloNumDims, true);
            x_model_backbone_b_5 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_b_5, [3 4 2 1], 4);

            x_model_backbone_b_5 = dlarray(single(x_model_backbone_b_5), 'SSCB');
        end

        function [x_model_backbone_b_5, x_model_backbone_b_5NumDims1002] = ReduceMeanGraph1000(this, x_model_backbone_blo, x_model_backbone_bloNumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1001, x_model_backbone_bloNumDims);
            xMean = mean(x_model_backbone_blo, dims);
            x_model_backbone_b_5 = xMean;
            x_model_backbone_b_5NumDims = x_model_backbone_bloNumDims;

            % Set graph output arguments
            x_model_backbone_b_5NumDims1002 = x_model_backbone_b_5NumDims;

        end

    end

end