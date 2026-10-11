classdef ReduceMeanLayer1007 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1007';
        end
    end


    methods
        function this = ReduceMeanLayer1007(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_105'};
        end

        function [x_model_backbone_105] = predict(this, x_model_backbone__99)
            if isdlarray(x_model_backbone__99)
                x_model_backbone__99 = stripdims(x_model_backbone__99);
            end
            x_model_backbone__99NumDims = 4;
            x_model_backbone__99 = netrai_fold0.ops.permuteInputVar(x_model_backbone__99, [4 3 1 2], 4);

            [x_model_backbone_105, x_model_backbone_105NumDims] = ReduceMeanGraph1021(this, x_model_backbone__99, x_model_backbone__99NumDims, false);
            x_model_backbone_105 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_105, [3 4 2 1], 4);

            x_model_backbone_105 = dlarray(single(x_model_backbone_105), 'SSCB');
        end

        function [x_model_backbone_105] = forward(this, x_model_backbone__99)
            if isdlarray(x_model_backbone__99)
                x_model_backbone__99 = stripdims(x_model_backbone__99);
            end
            x_model_backbone__99NumDims = 4;
            x_model_backbone__99 = netrai_fold0.ops.permuteInputVar(x_model_backbone__99, [4 3 1 2], 4);

            [x_model_backbone_105, x_model_backbone_105NumDims] = ReduceMeanGraph1021(this, x_model_backbone__99, x_model_backbone__99NumDims, true);
            x_model_backbone_105 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_105, [3 4 2 1], 4);

            x_model_backbone_105 = dlarray(single(x_model_backbone_105), 'SSCB');
        end

        function [x_model_backbone_105, x_model_backbone_105NumDims1023] = ReduceMeanGraph1021(this, x_model_backbone__99, x_model_backbone__99NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1022, x_model_backbone__99NumDims);
            xMean = mean(x_model_backbone__99, dims);
            x_model_backbone_105 = xMean;
            x_model_backbone_105NumDims = x_model_backbone__99NumDims;

            % Set graph output arguments
            x_model_backbone_105NumDims1023 = x_model_backbone_105NumDims;

        end

    end

end