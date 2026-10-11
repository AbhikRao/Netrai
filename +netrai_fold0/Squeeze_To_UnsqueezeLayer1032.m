classdef Squeeze_To_UnsqueezeLayer1032 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.Squeeze_To_UnsqueezeLayer1032';
        end
    end


    methods
        function this = Squeeze_To_UnsqueezeLayer1032(name)
            this.Name = name;
            this.OutputNames = {'x_Unsqueeze_output_0'};
        end

        function [x_Unsqueeze_output_0] = predict(this, x_model_ref_head_Gem)
            if isdlarray(x_model_ref_head_Gem)
                x_model_ref_head_Gem = stripdims(x_model_ref_head_Gem);
            end
            x_model_ref_head_GemNumDims = 2;
            x_model_ref_head_Gem = netrai_fold0.ops.permuteInputVar(x_model_ref_head_Gem, [2 1], 2);

            [x_Unsqueeze_output_0, x_Unsqueeze_output_0NumDims] = Squeeze_To_UnsqueezeGraph1096(this, x_model_ref_head_Gem, x_model_ref_head_GemNumDims, false);
            x_Unsqueeze_output_0 = netrai_fold0.ops.permuteOutputVar(x_Unsqueeze_output_0, [2 1], 2);

            x_Unsqueeze_output_0 = dlarray(single(x_Unsqueeze_output_0), 'CB');
        end

        function [x_Unsqueeze_output_0] = forward(this, x_model_ref_head_Gem)
            if isdlarray(x_model_ref_head_Gem)
                x_model_ref_head_Gem = stripdims(x_model_ref_head_Gem);
            end
            x_model_ref_head_GemNumDims = 2;
            x_model_ref_head_Gem = netrai_fold0.ops.permuteInputVar(x_model_ref_head_Gem, [2 1], 2);

            [x_Unsqueeze_output_0, x_Unsqueeze_output_0NumDims] = Squeeze_To_UnsqueezeGraph1096(this, x_model_ref_head_Gem, x_model_ref_head_GemNumDims, true);
            x_Unsqueeze_output_0 = netrai_fold0.ops.permuteOutputVar(x_Unsqueeze_output_0, [2 1], 2);

            x_Unsqueeze_output_0 = dlarray(single(x_Unsqueeze_output_0), 'CB');
        end

        function [x_Unsqueeze_output_0, x_Unsqueeze_output_0NumDims1097] = Squeeze_To_UnsqueezeGraph1096(this, x_model_ref_head_Gem, x_model_ref_head_GemNumDims, Training)

            % Execute the operators:
            % Squeeze:
            [x_model_Squeeze_outp, x_model_Squeeze_outpNumDims] = netrai_fold0.ops.onnxSqueeze(x_model_ref_head_Gem, this.Vars.x_model_Constant_out, x_model_ref_head_GemNumDims);

            % Unsqueeze:
            [shape, x_Unsqueeze_output_0NumDims] = netrai_fold0.ops.prepareUnsqueezeArgs(x_model_Squeeze_outp, this.Vars.x_Constant_output_0, x_model_Squeeze_outpNumDims);
            x_Unsqueeze_output_0 = reshape(x_model_Squeeze_outp, shape);

            % Set graph output arguments
            x_Unsqueeze_output_0NumDims1097 = x_Unsqueeze_output_0NumDims;

        end

    end

end