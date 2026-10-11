classdef ReduceMeanLayer1031 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1031';
        end
    end


    methods
        function this = ReduceMeanLayer1031(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_461'};
        end

        function [x_model_backbone_461] = predict(this, x_model_backbone_455)
            if isdlarray(x_model_backbone_455)
                x_model_backbone_455 = stripdims(x_model_backbone_455);
            end
            x_model_backbone_455NumDims = 4;
            x_model_backbone_455 = netrai_fold0.ops.permuteInputVar(x_model_backbone_455, [4 3 1 2], 4);

            [x_model_backbone_461, x_model_backbone_461NumDims] = ReduceMeanGraph1093(this, x_model_backbone_455, x_model_backbone_455NumDims, false);
            x_model_backbone_461 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_461, [3 4 2 1], 4);

            x_model_backbone_461 = dlarray(single(x_model_backbone_461), 'SSCB');
        end

        function [x_model_backbone_461] = forward(this, x_model_backbone_455)
            if isdlarray(x_model_backbone_455)
                x_model_backbone_455 = stripdims(x_model_backbone_455);
            end
            x_model_backbone_455NumDims = 4;
            x_model_backbone_455 = netrai_fold0.ops.permuteInputVar(x_model_backbone_455, [4 3 1 2], 4);

            [x_model_backbone_461, x_model_backbone_461NumDims] = ReduceMeanGraph1093(this, x_model_backbone_455, x_model_backbone_455NumDims, true);
            x_model_backbone_461 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_461, [3 4 2 1], 4);

            x_model_backbone_461 = dlarray(single(x_model_backbone_461), 'SSCB');
        end

        function [x_model_backbone_461, x_model_backbone_461NumDims1095] = ReduceMeanGraph1093(this, x_model_backbone_455, x_model_backbone_455NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1094, x_model_backbone_455NumDims);
            xMean = mean(x_model_backbone_455, dims);
            x_model_backbone_461 = xMean;
            x_model_backbone_461NumDims = x_model_backbone_455NumDims;

            % Set graph output arguments
            x_model_backbone_461NumDims1095 = x_model_backbone_461NumDims;

        end

    end

end