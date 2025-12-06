<template>
  <div class="dataset-page">
    <!-- 数据集列表视图 -->
    <template v-if="!currentView">
      <div class="page-header">
        <h1 class="gradient-text">数据集管理</h1>
        <p class="subtitle">{{ datasetsDir }}</p>
      </div>

      <!-- 操作栏 -->
      <div class="dataset-toolbar glass-card">
        <el-button type="primary" size="large" @click="showCreateDialog = true">
          <el-icon><Plus /></el-icon>
          新建数据集
        </el-button>
        <el-button size="large" @click="loadLocalDatasets">
          <el-icon><Refresh /></el-icon>
          刷新
        </el-button>
        <el-divider direction="vertical" />
        <el-divider direction="vertical" />
        <div class="toolbar-section">
          <input
            type="file"
            ref="folderInput"
            webkitdirectory
            directory
            hidden
            @change="handleFolderSelect"
          />
          <el-button type="primary" size="large" @click="triggerFolderUpload" :loading="isUploadingFolder">
            <el-icon><Upload /></el-icon>
            上传文件夹
          </el-button>
        </div>
      </div>

      <!-- 数据集文件夹网格 -->
      <div class="folder-grid" v-if="localDatasets.length > 0">
        <div 
          class="folder-card glass-card"
          v-for="ds in localDatasets"
          :key="ds.name"
          @click="openDataset(ds)"
        >
          <div class="folder-icon">
            <el-icon :size="48"><Folder /></el-icon>
          </div>
          <div class="folder-info">
            <div class="folder-name">{{ ds.name }}</div>
            <div class="folder-meta">{{ ds.imageCount }} 张图片</div>
          </div>
          <el-button 
            class="delete-btn"
            type="danger"
            :icon="Delete"
            circle
            size="small"
            @click.stop="confirmDeleteDataset(ds)"
          />
        </div>
      </div>

      <!-- 空状态 -->
      <div class="empty-state glass-card" v-else>
        <el-icon :size="64"><FolderOpened /></el-icon>
        <h3>暂无数据集</h3>
        <p>点击「新建数据集」创建第一个数据集</p>
      </div>
    </template>

    <!-- 数据集详情视图 -->
    <template v-else>
      <!-- 顶部导航栏 -->
      <div class="detail-header glass-card">
        <div class="header-left">
          <el-button @click="goBack" class="back-btn">
            <el-icon><ArrowLeft /></el-icon>
          </el-button>
          <div class="header-info">
            <h2>{{ currentView.name }}</h2>
            <span class="path-text">{{ currentView.path }}</span>
          </div>
        </div>
        <div class="header-right">
          <el-upload
            :http-request="customUpload"
            :multiple="true"
            :show-file-list="false"
            :before-upload="beforeUpload"
            accept="image/*,.txt,.safetensors"
          >
            <el-button type="primary" :loading="isUploading">
              <el-icon><Upload /></el-icon>
              上传文件
            </el-button>
          </el-upload>
        </div>
      </div>

      <!-- 数据集统计 -->
      <div class="dataset-info glass-card" v-if="datasetStore.currentDataset">
      <div class="info-header">
        <div class="info-stats">
          <div class="stat">
            <el-icon><Picture /></el-icon>
            <span>{{ datasetStore.currentDataset.imageCount }} 张图片</span>
          </div>
          <div class="stat">
            <el-icon><Folder /></el-icon>
            <span>{{ formatSize(datasetStore.currentDataset.totalSize) }}</span>
          </div>
          <div class="stat" :class="{ 'stat-success': latentCachedCount === datasetStore.currentDataset.imageCount }">
            <el-icon><Box /></el-icon>
            <span>Latent: {{ latentCachedCount }} / {{ datasetStore.currentDataset.imageCount }}</span>
          </div>
          <div class="stat" :class="{ 'stat-success': textCachedCount === datasetStore.currentDataset.imageCount }">
            <el-icon><Document /></el-icon>
            <span>Text: {{ textCachedCount }} / {{ datasetStore.currentDataset.imageCount }}</span>
          </div>
        </div>
        <div class="info-actions">
          <el-button @click="toggleSelectAll" size="small">
            {{ isAllSelected ? '取消全选' : '全选' }}
          </el-button>
          <el-button type="danger" size="small" @click="deleteSelected" :disabled="datasetStore.selectedImages.size === 0">
            <el-icon><Delete /></el-icon>
            删除 ({{ datasetStore.selectedImages.size }})
          </el-button>
          <el-button type="primary" size="small" @click="generateCache" :loading="isGeneratingCache">
            <el-icon><Box /></el-icon>
            一键生成缓存
          </el-button>
          <el-button type="danger" size="small" @click="showClearCacheDialog = true">
            <el-icon><Delete /></el-icon>
            清理缓存
          </el-button>
          <el-button type="warning" size="small" @click="showOllamaDialog = true">
            <el-icon><MagicStick /></el-icon>
            Ollama 标注
          </el-button>
          <el-button type="info" size="small" @click="showResizeDialog = true">
            <el-icon><ScaleToOriginal /></el-icon>
            图片缩放
          </el-button>
          <el-button type="danger" size="small" @click="confirmDeleteCaptions" plain>
            <el-icon><Delete /></el-icon>
            删除标注
          </el-button>
          <el-button type="success" size="small" @click="showBucketCalculator = true">
            <el-icon><Grid /></el-icon>
            分桶计算器
          </el-button>
        </div>
      </div>
      
      <!-- 缓存生成进度条 -->
      <div class="cache-progress-section" v-if="cacheStatus.latent.status === 'running' || cacheStatus.text.status === 'running'">
        <div class="cache-progress-item" v-if="cacheStatus.latent.status === 'running'">
          <div class="progress-label">
            <el-icon class="spinning"><Loading /></el-icon>
            <span>Latent 缓存</span>
            <span class="progress-count" v-if="cacheStatus.latent.current && cacheStatus.latent.total">
              {{ cacheStatus.latent.current }} / {{ cacheStatus.latent.total }}
            </span>
          </div>
          <el-progress 
            :percentage="cacheStatus.latent.progress || 0" 
            :stroke-width="8"
            color="#f0b429"
          />
        </div>
        <div class="cache-progress-item" v-if="cacheStatus.text.status === 'running'">
          <div class="progress-label">
            <el-icon class="spinning"><Loading /></el-icon>
            <span>Text 缓存</span>
            <span class="progress-count" v-if="cacheStatus.text.current && cacheStatus.text.total">
              {{ cacheStatus.text.current }} / {{ cacheStatus.text.total }}
            </span>
          </div>
          <el-progress 
            :percentage="cacheStatus.text.progress || 0" 
            :stroke-width="8"
            color="#67c23a"
          />
        </div>
        <div class="cache-progress-item queued" v-if="cacheStatus.latent.status === 'running' && cacheStatus.text.status !== 'running' && isGeneratingCache">
          <div class="progress-label">
            <el-icon><Clock /></el-icon>
            <span>Text 缓存（排队中，等待 Latent 完成）</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 图片网格 -->
    <div class="image-grid" v-if="datasetStore.currentImages.length > 0">
      <div 
        class="image-card glass-card"
        v-for="image in datasetStore.currentImages"
        :key="image.path"
        :class="{ selected: datasetStore.selectedImages.has(image.path) }"
      >
        <div class="image-wrapper" @click="previewImage(image)">
          <img 
            :src="getImageUrl(image)" 
            :alt="image.filename" 
            loading="lazy"
            @error="handleImageError($event, image)"
            :data-retry="imageRetryCount.get(image.path) || 0"
          />
          <!-- 加载失败占位 -->
          <div class="image-error-overlay" v-if="imageLoadFailed.has(image.path)">
            <el-icon><WarningFilled /></el-icon>
            <span>加载失败</span>
            <el-button size="small" @click.stop="retryLoadImage(image)">重试</el-button>
          </div>
          <!-- 选择圆圈 -->
          <div 
            class="select-circle"
            :class="{ checked: datasetStore.selectedImages.has(image.path) }"
            @click.stop="toggleSelection(image)"
          >
            <el-icon v-if="datasetStore.selectedImages.has(image.path)"><Check /></el-icon>
          </div>
          <!-- 缓存状态标签 -->
          <div class="cache-tags">
            <div class="cache-tag" :class="{ active: image.hasLatentCache }" title="Latent缓存">
              <el-icon><Box /></el-icon>
              <span>L</span>
            </div>
            <div class="cache-tag" :class="{ active: image.hasTextCache }" title="Text缓存">
              <el-icon><Document /></el-icon>
              <span>T</span>
            </div>
          </div>
        </div>
        <div class="image-info">
          <div class="image-name" :title="image.filename">{{ image.filename }}</div>
          <div class="image-meta">
            {{ image.width }}×{{ image.height }} · {{ formatSize(image.size) }}
          </div>
          <div class="image-caption" :class="{ 'no-caption': !image.caption }">
            {{ image.caption || '无标注' }}
          </div>
        </div>
        <div class="image-actions">
          <!-- 编辑按钮已移除，直接点击图片即可 -->
        </div>
      </div>
    </div>

    <!-- 加载中状态 -->
    <div class="loading-state glass-card" v-else-if="datasetStore.isLoading">
      <el-icon :size="64" class="is-loading"><Loading /></el-icon>
      <h3>正在加载数据集...</h3>
      <p>请稍候，正在扫描图片和缓存信息</p>
    </div>

    <!-- 空状态 -->
    <div class="empty-state glass-card" v-else>
      <el-icon :size="64"><FolderOpened /></el-icon>
      <h3>暂无图片</h3>
      <p>上传图片到数据集</p>
    </div>
    </template>

    <!-- 统一的图片预览与编辑对话框 -->
    <el-dialog
      v-model="previewDialogVisible"
      title="图片预览与编辑"
      width="1200px"
      class="preview-edit-dialog"
      :close-on-click-modal="true"
      align-center
    >
      <div class="preview-edit-layout" v-if="editingImage">
        <!-- 左侧：图片预览 -->
        <div class="preview-side">
          <div class="image-wrapper">
            <img :src="`/api/dataset/image?path=${encodeURIComponent(editingImage.path)}`" :alt="editingImage.filename" />
          </div>
          <div class="image-meta-info">
            <span>{{ editingImage.width }} x {{ editingImage.height }}</span>
            <span>{{ formatSize(editingImage.size) }}</span>
            <span>{{ editingImage.filename }}</span>
          </div>
        </div>
        
        <!-- 右侧：标注编辑 -->
        <div class="edit-side">
          <div class="edit-header">
            <h3>图片标注</h3>
            <div class="edit-actions">
               <el-button type="primary" @click="saveCaption" :loading="isSavingCaption">
                保存标注
              </el-button>
            </div>
          </div>
          
          <el-input
            v-model="editingCaption"
            type="textarea"
            :rows="20"
            placeholder="输入图片描述..."
            resize="none"
            class="caption-textarea"
          />
          
          <div class="keyboard-hint">
            <el-icon><InfoFilled /></el-icon>
            <span>提示: 支持 Ctrl+Enter 快速保存</span>
          </div>
        </div>
      </div>
    </el-dialog>

    <!-- 批量生成标注对话框 -->
    <el-dialog
      v-model="showCaptionDialog"
      title="批量生成标注"
      width="500px"
    >
      <el-form>
        <el-form-item label="模型">
          <el-select v-model="captionModel" style="width: 100%">
            <el-option label="Qwen-VL (推荐)" value="qwen" />
            <el-option label="BLIP-2" value="blip" />
          </el-select>
        </el-form-item>
        <el-form-item label="提示词">
          <el-input
            v-model="captionPrompt"
            type="textarea"
            :rows="3"
            placeholder="描述这张图片..."
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showCaptionDialog = false">取消</el-button>
        <el-button type="primary" @click="generateCaptions" :loading="isGenerating">
          开始生成
        </el-button>
      </template>
    </el-dialog>

    <!-- 新建数据集对话框 -->
    <el-dialog
      v-model="showCreateDialog"
      title="新建数据集"
      width="400px"
    >
      <el-form>
        <el-form-item label="数据集名称">
          <el-input
            v-model="newDatasetName"
            placeholder="输入数据集名称..."
            @keyup.enter="createDataset"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showCreateDialog = false">取消</el-button>
        <el-button type="primary" @click="createDataset" :loading="isCreating">
          创建
        </el-button>
      </template>
    </el-dialog>

    <!-- 生成缓存对话框 -->
    <el-dialog
      v-model="showCacheDialog"
      title="生成缓存"
      width="500px"
    >
      <el-form label-width="auto">
        <el-form-item label="选择缓存类型">
          <el-checkbox-group v-model="cacheOptions">
            <el-checkbox label="latent">
              Latent 缓存
              <span class="cache-path-hint" v-if="trainingStore.config.vaePath">({{ trainingStore.config.vaePath.split(/[/\\]/).pop() }})</span>
              <span class="cache-path-missing" v-else>(未配置VAE)</span>
            </el-checkbox>
            <el-checkbox label="text">
              Text 缓存
              <span class="cache-path-hint" v-if="trainingStore.config.textEncoderPath">({{ trainingStore.config.textEncoderPath.split(/[/\\]/).pop() }})</span>
              <span class="cache-path-missing" v-else>(未配置Text Encoder)</span>
            </el-checkbox>
          </el-checkbox-group>
        </el-form-item>
      </el-form>
      
      <div class="cache-warning" v-if="!hasRequiredPaths">
        <el-icon><WarningFilled /></el-icon>
        <span>请先在「训练配置」页面设置模型路径</span>
        <el-button type="primary" link @click="goToConfig">前往配置</el-button>
      </div>
      
      <div class="cache-hint" v-else>
        <el-icon><InfoFilled /></el-icon>
        <span>缓存文件将保存在数据集目录中</span>
      </div>
      
      <template #footer>
        <el-button @click="showCacheDialog = false">取消</el-button>
        <el-button type="primary" @click="confirmGenerateCache" :loading="isGeneratingCache" :disabled="!canGenerateCache">
          开始生成
        </el-button>
      </template>
    </el-dialog>
    <!-- 缓存清理对话框 -->
    <el-dialog
      v-model="showClearCacheDialog"
      title="清理缓存"
      width="500px"
    >
      <el-form label-position="top">
        <el-form-item label="选择清理类型">
          <div class="flex flex-col gap-2">
            <el-checkbox v-model="clearCacheOptions.latent">Latent 缓存</el-checkbox>
            <el-checkbox v-model="clearCacheOptions.text">Text 缓存</el-checkbox>
          </div>
        </el-form-item>
        <el-alert
          title="清理后需要重新生成才能用于训练"
          type="warning"
          :closable="false"
          show-icon
        />
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showClearCacheDialog = false">取消</el-button>
          <el-button 
            type="danger" 
            @click="startClearCache"
            :loading="isClearingCache"
            :disabled="!clearCacheOptions.latent && !clearCacheOptions.text"
          >
            确认清理
          </el-button>
        </span>
      </template>
    </el-dialog>

    <!-- 图片缩放对话框 -->
    <el-dialog
      v-model="showResizeDialog"
      title="批量缩放图片"
      width="500px"
    >
      <el-alert
        title="⚠️ 此操作不可撤销！"
        type="error"
        description="图片将被直接覆盖，原图无法恢复。建议先备份数据集。"
        :closable="false"
        show-icon
        style="margin-bottom: 20px"
      />
      
      <el-form label-width="100px" :disabled="resizing">
        <el-form-item label="长边尺寸">
          <el-slider v-model="resizeConfig.maxLongEdge" :min="512" :max="2048" :step="64" show-input />
          <div class="form-hint">大于此尺寸的图片将被缩放</div>
        </el-form-item>
        
        <el-form-item label="JPEG 质量">
          <el-slider v-model="resizeConfig.quality" :min="70" :max="100" :step="5" show-input />
        </el-form-item>
        
        <el-form-item label="锐化强度">
          <el-slider v-model="resizeConfig.sharpen" :min="0" :max="1" :step="0.1" show-input />
          <div class="form-hint">缩放后锐化恢复细节 (推荐 0.3)</div>
        </el-form-item>
      </el-form>
      
      <div class="resize-hint">
        <el-icon><InfoFilled /></el-icon>
        <span>使用高质量多步下采样 + USM锐化算法</span>
      </div>
      
      <!-- 进度显示 -->
      <div class="resize-progress" v-if="resizing || resizeStatus.completed > 0">
        <el-progress 
          :percentage="resizeProgress" 
          :status="resizeStatus.running ? '' : 'success'"
        />
        <div class="progress-info">
          <span>{{ resizeStatus.completed }} / {{ resizeStatus.total }}</span>
          <span v-if="resizeStatus.current_file">正在处理: {{ resizeStatus.current_file }}</span>
        </div>
      </div>
      
      <template #footer>
        <el-button @click="showResizeDialog = false" :disabled="resizing">关闭</el-button>
        <el-button v-if="resizing" type="danger" @click="stopResize">
          停止
        </el-button>
        <el-button v-else type="danger" @click="confirmResize">
          确认缩放
        </el-button>
      </template>
    </el-dialog>

    <!-- Ollama 标注对话框 -->
    <el-dialog
      v-model="showOllamaDialog"
      title="Ollama 图片标注"
      width="600px"
      @open="loadOllamaModels"
    >
      <el-form label-width="100px" :disabled="ollamaTagging">
        <el-form-item label="Ollama 地址">
          <div class="url-input-row">
            <el-input v-model="ollamaConfig.url" placeholder="http://localhost:11434" />
            <el-button @click="testOllamaConnection" :loading="testingConnection">
              测试
            </el-button>
          </div>
        </el-form-item>
        
        <el-form-item label="模型">
          <el-select v-model="ollamaConfig.model" placeholder="选择模型" style="width: 100%">
            <el-option v-for="m in ollamaModels" :key="m" :label="m" :value="m" />
          </el-select>
          <div class="form-hint" v-if="ollamaModels.length === 0">
            请先测试连接以获取模型列表
          </div>
        </el-form-item>
        
        <el-form-item label="长边尺寸">
          <el-slider v-model="ollamaConfig.maxLongEdge" :min="512" :max="2048" :step="64" show-input />
          <div class="form-hint">图片将被缩放到此尺寸再发送给 Ollama</div>
        </el-form-item>
        
        <el-form-item label="提示词">
          <el-input
            v-model="ollamaConfig.prompt"
            type="textarea"
            :rows="6"
            placeholder="描述这张图片..."
          />
        </el-form-item>
        
        <el-form-item label="跳过已有">
          <el-switch v-model="ollamaConfig.skipExisting" />
          <span class="switch-label">跳过已有 .txt 标注的图片</span>
        </el-form-item>
        
        <el-form-item label="触发词">
          <el-input 
            v-model="ollamaConfig.triggerWord" 
            placeholder="如: zst_style, my_character"
            clearable
          />
          <div class="form-hint">将此词添加到所有标注开头，用于 LoRA 训练触发</div>
        </el-form-item>
        
        <el-form-item label="思考模式">
          <el-switch v-model="ollamaConfig.enableThink" />
          <span class="switch-label">启用模型思考（qwen3-vl 等需要开启）</span>
          <div class="form-hint">💡 qwen3-vl 默认需要开启；llava/moondream 等传统模型请关闭</div>
        </el-form-item>
      </el-form>
      
      <!-- 进度显示 -->
      <div class="ollama-progress" v-if="ollamaTagging || ollamaStatus.completed > 0">
        <el-progress 
          :percentage="ollamaProgress" 
          :status="ollamaStatus.running ? '' : 'success'"
        />
        <div class="progress-info">
          <span>{{ ollamaStatus.completed }} / {{ ollamaStatus.total }}</span>
          <span v-if="ollamaStatus.current_file">正在处理: {{ ollamaStatus.current_file }}</span>
          <span v-if="ollamaStatus.errors.length > 0" class="error-count">
            失败: {{ ollamaStatus.errors.length }}
          </span>
        </div>
      </div>
      
      <template #footer>
        <el-button @click="showOllamaDialog = false" :disabled="ollamaTagging">关闭</el-button>
        <el-button v-if="ollamaTagging" type="danger" @click="stopOllamaTagging">
          停止标注
        </el-button>
        <el-button v-else type="primary" @click="startOllamaTagging" :disabled="!canStartOllama">
          开始标注
        </el-button>
      </template>
    </el-dialog>

    <!-- 分桶计算器对话框 -->
    <el-dialog
      v-model="showBucketCalculator"
      title="分桶计算器"
      width="800px"
      class="bucket-dialog"
    >
      <div class="bucket-config">
        <el-form :inline="true" label-width="100px">
          <el-form-item label="Batch Size">
            <el-input-number v-model="bucketConfig.batchSize" :min="1" :max="16" />
          </el-form-item>
          <el-form-item label="分辨率限制">
            <el-input-number v-model="bucketConfig.resolutionLimit" :min="256" :max="2048" :step="64" />
          </el-form-item>
          <el-form-item>
            <el-button type="primary" @click="calculateBuckets" :loading="calculatingBuckets">
              计算分桶
            </el-button>
          </el-form-item>
        </el-form>
      </div>
      
      <div class="bucket-results" v-if="bucketResults.length > 0">
        <div class="bucket-summary">
          <div class="summary-item">
            <span class="label">总图片数</span>
            <span class="value">{{ bucketSummary.totalImages }}</span>
          </div>
          <div class="summary-item">
            <span class="label">桶数量</span>
            <span class="value">{{ bucketResults.length }}</span>
          </div>
          <div class="summary-item">
            <span class="label">总批次数</span>
            <span class="value">{{ bucketSummary.totalBatches }}</span>
          </div>
          <div class="summary-item">
            <span class="label">丢弃图片</span>
            <span class="value" :class="{ 'text-warning': bucketSummary.droppedImages > 0 }">
              {{ bucketSummary.droppedImages }}
            </span>
          </div>
        </div>
        
        <el-table :data="bucketResults" style="width: 100%" max-height="400">
          <el-table-column prop="resolution" label="分辨率" width="120">
            <template #default="{ row }">
              {{ row.width }}×{{ row.height }}
            </template>
          </el-table-column>
          <el-table-column prop="aspectRatio" label="宽高比" width="100">
            <template #default="{ row }">
              {{ row.aspectRatio.toFixed(2) }}
            </template>
          </el-table-column>
          <el-table-column prop="count" label="图片数" width="80" />
          <el-table-column prop="batches" label="批次数" width="80" />
          <el-table-column prop="dropped" label="丢弃" width="60">
            <template #default="{ row }">
              <span :class="{ 'text-warning': row.dropped > 0 }">{{ row.dropped }}</span>
            </template>
          </el-table-column>
          <el-table-column label="分布" min-width="200">
            <template #default="{ row }">
              <el-progress 
                :percentage="row.percentage" 
                :stroke-width="12"
                :show-text="false"
                :color="getBucketColor(row.aspectRatio)"
              />
            </template>
          </el-table-column>
        </el-table>
      </div>
      
      <div class="bucket-empty" v-else-if="!calculatingBuckets">
        <el-icon :size="48"><Grid /></el-icon>
        <p>点击「计算分桶」查看数据集的分桶分布</p>
      </div>
      
      <template #footer>
        <el-button @click="showBucketCalculator = false">关闭</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useDatasetStore, type DatasetImage } from '@/stores/dataset'
import { useTrainingStore } from '@/stores/training'
import { useWebSocketStore } from '@/stores/websocket'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete, InfoFilled, WarningFilled, MagicStick, ScaleToOriginal, Loading, Clock, Grid } from '@element-plus/icons-vue'
import axios from 'axios'

const datasetStore = useDatasetStore()
const trainingStore = useTrainingStore()
const wsStore = useWebSocketStore()

// 图片加载失败处理
const imageLoadFailed = ref(new Set<string>())
const imageRetryCount = ref(new Map<string, number>())
const MAX_RETRY = 2  // 减少自动重试次数

// 获取图片URL（带时间戳防止缓存问题）
function getImageUrl(image: DatasetImage): string {
  const retry = imageRetryCount.value.get(image.path) || 0
  // 每次重试都添加不同的时间戳来绕过缓存
  const cacheBuster = retry > 0 ? `&_t=${Date.now()}&_r=${retry}` : ''
  return `${image.thumbnailUrl}${cacheBuster}`
}

// 图片加载失败处理
function handleImageError(event: Event, image: DatasetImage) {
  const retryCount = imageRetryCount.value.get(image.path) || 0
  console.log(`[Image] Load failed: ${image.filename}, retry: ${retryCount}/${MAX_RETRY}`)
  
  if (retryCount < MAX_RETRY) {
    // 自动重试
    const newRetry = retryCount + 1
    imageRetryCount.value.set(image.path, newRetry)
    const img = event.target as HTMLImageElement
    // 强制重新加载，递增延迟
    setTimeout(() => {
      const newUrl = getImageUrl(image)
      console.log(`[Image] Retrying: ${newUrl}`)
      img.src = newUrl
    }, 1000 * newRetry) // 1s, 2s 延迟
  } else {
    // 重试次数用完，标记为失败
    console.log(`[Image] Max retries reached: ${image.filename}`)
    imageLoadFailed.value.add(image.path)
  }
}

// 手动重试加载图片（重置重试计数，强制刷新）
function retryLoadImage(image: DatasetImage) {
  console.log(`[Image] Manual retry: ${image.filename}`)
  imageLoadFailed.value.delete(image.path)
  imageRetryCount.value.delete(image.path)  // 完全重置
  // 强制触发响应式更新
  imageRetryCount.value = new Map(imageRetryCount.value)
  imageLoadFailed.value = new Set(imageLoadFailed.value)
}

// 缓存状态（从 WebSocket 获取实时进度）
const cacheStatus = computed(() => wsStore.cacheStatus)

// 视图状态
interface LocalDataset {
  name: string
  path: string
  imageCount: number
}

const currentView = ref<LocalDataset | null>(null)
const localDatasets = ref<LocalDataset[]>([])
const datasetsDir = ref('')
const datasetPath = ref('')

const latentCachedCount = computed(() => {
  return datasetStore.currentImages.filter(img => img.hasLatentCache).length
})

const textCachedCount = computed(() => {
  return datasetStore.currentImages.filter(img => img.hasTextCache).length
})

// 对话框状态
const captionDialogVisible = ref(false) // Deprecated, kept for safety or remove
const previewDialogVisible = ref(false)
const showCaptionDialog = ref(false)
const showCreateDialog = ref(false)
const editingImage = ref<DatasetImage | null>(null)
const editingCaption = ref('')
// const previewImageUrl = ref('') // Removed
const captionModel = ref('qwen')
const captionPrompt = ref('详细描述这张图片的内容、风格和氛围')
const isGenerating = ref(false)
const isGeneratingCache = ref(false)
const showClearCacheDialog = ref(false)
const isClearingCache = ref(false)
const clearCacheOptions = ref({
  latent: true,
  text: true
})

// 图片缩放相关
const showResizeDialog = ref(false)
const resizing = ref(false)
const resizeConfig = ref({
  maxLongEdge: 1024,
  quality: 95,
  sharpen: 0.3
})
const resizeStatus = ref({
  running: false,
  total: 0,
  completed: 0,
  current_file: ''
})

// 分桶计算器相关
const showBucketCalculator = ref(false)
const calculatingBuckets = ref(false)
const bucketConfig = ref({
  batchSize: 4,
  resolutionLimit: 1536
})
interface BucketInfo {
  width: number
  height: number
  aspectRatio: number
  count: number
  batches: number
  dropped: number
  percentage: number
}
const bucketResults = ref<BucketInfo[]>([])
const bucketSummary = computed(() => {
  const totalImages = bucketResults.value.reduce((sum, b) => sum + b.count, 0)
  const totalBatches = bucketResults.value.reduce((sum, b) => sum + b.batches, 0)
  const droppedImages = bucketResults.value.reduce((sum, b) => sum + b.dropped, 0)
  return { totalImages, totalBatches, droppedImages }
})

function getBucketColor(aspectRatio: number): string {
  // 根据宽高比返回颜色
  if (aspectRatio < 0.8) return '#67c23a' // 竖图 - 绿色
  if (aspectRatio > 1.2) return '#409eff' // 横图 - 蓝色
  return '#f0b429' // 方图 - 金色
}

async function calculateBuckets() {
  if (!currentView.value) return
  
  calculatingBuckets.value = true
  bucketResults.value = []
  
  try {
    // 从当前图片列表计算分桶
    const images = datasetStore.currentImages
    const limit = bucketConfig.value.resolutionLimit
    const batchSize = bucketConfig.value.batchSize
    
    // 按分辨率分组
    const buckets: Record<string, { width: number; height: number; count: number }> = {}
    
    for (const img of images) {
      let w = img.width
      let h = img.height
      
      // 应用分辨率限制
      if (Math.max(w, h) > limit) {
        const scale = limit / Math.max(w, h)
        w = Math.floor(w * scale)
        h = Math.floor(h * scale)
      }
      
      // 对齐到 8 的倍数
      w = Math.floor(w / 8) * 8
      h = Math.floor(h / 8) * 8
      
      const key = `${w}x${h}`
      if (!buckets[key]) {
        buckets[key] = { width: w, height: h, count: 0 }
      }
      buckets[key].count++
    }
    
    // 计算每个桶的批次数和丢弃数
    const results: BucketInfo[] = []
    const maxCount = Math.max(...Object.values(buckets).map(b => b.count))
    
    for (const [key, bucket] of Object.entries(buckets)) {
      const batches = Math.floor(bucket.count / batchSize)
      const dropped = bucket.count % batchSize
      
      results.push({
        width: bucket.width,
        height: bucket.height,
        aspectRatio: bucket.width / bucket.height,
        count: bucket.count,
        batches,
        dropped: batches > 0 ? dropped : bucket.count, // 如果没有完整批次，全部丢弃
        percentage: Math.round((bucket.count / maxCount) * 100)
      })
    }
    
    // 按图片数量排序
    results.sort((a, b) => b.count - a.count)
    bucketResults.value = results
    
  } catch (error: any) {
    ElMessage.error('计算分桶失败: ' + error.message)
  } finally {
    calculatingBuckets.value = false
  }
}

// Ollama 标注相关
const showOllamaDialog = ref(false)
const ollamaModels = ref<string[]>([])
const testingConnection = ref(false)
const ollamaTagging = ref(false)
const ollamaConfig = ref({
  url: 'http://localhost:11434',
  model: '',
  prompt: `你是一位专门为 AI 绘画模型训练服务的打标专家。请为这张图片生成训练标注。

规则：
1. 使用中文短语/Tag 格式，用逗号分隔
2. 描述主体特征：人物、衣着、动作、物品等
3. 不要描述光影、背景、构图、风格
4. 简洁明了，不要写长句

示例输出：1个女孩, 黑发, 齐肩发, 白色连衣裙, 手摸脸, 微笑`,
  maxLongEdge: 1024,
  skipExisting: true,
  triggerWord: '',  // 触发词，添加到每个标注开头
  enableThink: true  // 默认开启思考模式（qwen3-vl 等模型需要）
})
const ollamaStatus = ref({
  running: false,
  total: 0,
  completed: 0,
  current_file: '',
  errors: [] as string[]
})

const newDatasetName = ref('')
const isCreating = ref(false)
const isSavingCaption = ref(false)

// ... (keep loadLocalDatasets and openDataset)

// 预览图片 (现在也是编辑入口)
function previewImage(image: DatasetImage) {
  editingImage.value = image
  editingCaption.value = image.caption || ''
  previewDialogVisible.value = true
}

// 编辑标注 (已废弃，保留兼容性或直接移除调用)
function editCaption(image: DatasetImage) {
  previewImage(image)
}

// 保存标注
async function saveCaption() {
  if (!editingImage.value) return
  
  isSavingCaption.value = true
  try {
    await datasetStore.updateCaption(editingImage.value.path, editingCaption.value)
    ElMessage.success('标注已保存')
    // previewDialogVisible.value = false // Optional: keep open to continue editing? User usually prefers staying or closing manually. Let's keep it open for now as per "Editor" feel.
  } catch (error) {
    ElMessage.error('保存失败')
  } finally {
    isSavingCaption.value = false
  }
}

// 缓存生成配置
const showCacheDialog = ref(false)
const cacheOptions = ref<string[]>(['latent', 'text'])

// 加载数据集列表
async function loadLocalDatasets() {
  try {
    const response = await axios.get('/api/dataset/list')
    localDatasets.value = response.data.datasets
    datasetsDir.value = response.data.datasetsDir
  } catch (error) {
    console.error('Failed to load datasets:', error)
  }
}

// 打开数据集
async function openDataset(ds: LocalDataset) {
  currentView.value = ds
  datasetPath.value = ds.path
  await datasetStore.scanDataset(ds.path)
}

// 返回列表
function goBack() {
  currentView.value = null
  datasetStore.currentDataset = null
}

const folderInput = ref<HTMLInputElement | null>(null)
const isUploadingFolder = ref(false)

// 触发文件夹选择
function triggerFolderUpload() {
  folderInput.value?.click()
}

// 处理文件夹选择
async function handleFolderSelect(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files || input.files.length === 0) return
  
  const files = Array.from(input.files)
  // 过滤非图片文件
  const validFiles = files.filter(f => 
    f.type.startsWith('image/') || 
    f.name.endsWith('.txt') || 
    f.name.endsWith('.safetensors')
  )
  
  if (validFiles.length === 0) {
    ElMessage.warning('未找到有效的图片或标注文件')
    input.value = ''
    return
  }
  
  // 默认使用文件夹名称
  const folderName = validFiles[0].webkitRelativePath.split('/')[0] || 'New Dataset'
  
  try {
    const { value: name } = await ElMessageBox.prompt('请输入数据集名称', '上传文件夹', {
      confirmButtonText: '开始上传',
      cancelButtonText: '取消',
      inputValue: folderName,
      inputValidator: (val) => !!val.trim() || '名称不能为空'
    })
    
    if (name) {
      await uploadFilesInBatches(validFiles, name)
    }
  } catch {
    // Cancelled
    input.value = ''
  }
}

// 分批上传文件
async function uploadFilesInBatches(files: File[], datasetName: string) {
  isUploadingFolder.value = true
  const batchSize = 20
  let successCount = 0
  let failCount = 0
  
  try {
    const totalBatches = Math.ceil(files.length / batchSize)
    
    for (let i = 0; i < files.length; i += batchSize) {
      const batch = files.slice(i, i + batchSize)
      const formData = new FormData()
      formData.append('dataset_name', datasetName)
      batch.forEach(f => formData.append('files', f))
      
      try {
        const res = await axios.post('/api/dataset/upload_batch', formData)
        successCount += res.data.uploaded
        if (res.data.errors) failCount += res.data.errors.length
      } catch (e) {
        failCount += batch.length
        console.error(e)
      }
      
      // Optional: Update progress message
      if (i + batchSize < files.length) {
        ElMessage.info(`正在上传: ${Math.round(((i + batchSize) / files.length) * 100)}%`)
      }
    }
    
    ElMessage.success(`上传完成: 成功 ${successCount}, 失败 ${failCount}`)
    await loadLocalDatasets()
    
    // Auto open if created
    const newDs = localDatasets.value.find(d => d.name === datasetName)
    if (newDs) openDataset(newDs)
    
  } catch (error: any) {
    ElMessage.error('上传过程中发生错误: ' + error.message)
  } finally {
    isUploadingFolder.value = false
    if (folderInput.value) folderInput.value.value = ''
  }
}

// 检查是否有必要的路径配置
const hasRequiredPaths = computed(() => {
  const needLatent = cacheOptions.value.includes('latent')
  const needText = cacheOptions.value.includes('text')
  
  if (needLatent && !trainingStore.config.vaePath) return false
  if (needText && !trainingStore.config.textEncoderPath) return false
  return true
})

const canGenerateCache = computed(() => {
  return cacheOptions.value.length > 0 && hasRequiredPaths.value
})

// 跳转到配置页面
function goToConfig() {
  showCacheDialog.value = false
  window.location.href = '/config'
}

// 打开缓存生成对话框
function generateCache() {
  if (!currentView.value) return
  showCacheDialog.value = true
}

// 确认生成缓存
async function confirmGenerateCache() {
  if (!currentView.value) return
  
  isGeneratingCache.value = true
  try {
    const response = await axios.post('/api/cache/generate', {
      datasetPath: currentView.value.path,
      generateLatent: cacheOptions.value.includes('latent'),
      generateText: cacheOptions.value.includes('text'),
      vaePath: trainingStore.config.vaePath,
      textEncoderPath: trainingStore.config.textEncoderPath
    })
    
    ElMessage.success('缓存生成任务已启动')
    showCacheDialog.value = false
    
    // 定期刷新数据集以更新缓存状态
    const refreshInterval = setInterval(async () => {
      if (currentView.value) {
        await datasetStore.scanDataset(currentView.value.path)
      }
    }, 3000)
    
    // 30秒后停止刷新
    setTimeout(() => {
      clearInterval(refreshInterval)
      isGeneratingCache.value = false
    }, 30000)
    
  } catch (error: any) {
    ElMessage.error('启动失败: ' + (error.response?.data?.detail || error.message))
    isGeneratingCache.value = false
  }
}

// 缓存清理
async function startClearCache() {
  if (!currentView.value) return
  
  try {
    isClearingCache.value = true
    const response = await axios.post('/api/cache/clear', {
      datasetPath: currentView.value.path,
      clearLatent: clearCacheOptions.value.latent,
      clearText: clearCacheOptions.value.text
    })
    
    const { deleted, errors } = response.data
    if (errors && errors.length > 0) {
      ElMessage.warning(`清理完成，但有 ${errors.length} 个文件失败`)
      console.error('Clear cache errors:', errors)
    } else {
      ElMessage.success(`成功清理 ${deleted} 个缓存文件`)
    }
    
    showClearCacheDialog.value = false
    // 刷新当前数据集
    await datasetStore.scanDataset(currentView.value.path)
  } catch (error: any) {
    ElMessage.error('清理失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    isClearingCache.value = false
  }
}


// 删除数据集确认
async function confirmDeleteDataset(ds: LocalDataset) {
  try {
    await ElMessageBox.confirm(
      `确定要删除数据集「${ds.name}」吗？此操作不可恢复！`,
      '删除确认',
      {
        confirmButtonText: '删除',
        cancelButtonText: '取消',
        type: 'warning',
        confirmButtonClass: 'el-button--danger'
      }
    )
    await deleteDataset(ds)
  } catch {
    // 用户取消
  }
}

// 删除数据集
async function deleteDataset(ds: LocalDataset) {
  try {
    await axios.delete(`/api/dataset/${encodeURIComponent(ds.name)}`)
    ElMessage.success(`数据集「${ds.name}」已删除`)
    await loadLocalDatasets()
  } catch (error: any) {
    ElMessage.error(error.response?.data?.detail || '删除失败')
  }
}

async function createDataset() {
  if (!newDatasetName.value.trim()) {
    ElMessage.warning('请输入数据集名称')
    return
  }
  
  isCreating.value = true
  try {
    const formData = new FormData()
    formData.append('name', newDatasetName.value.trim())
    const response = await axios.post('/api/dataset/create', formData)
    ElMessage.success(`数据集「${response.data.name}」创建成功`)
    await loadLocalDatasets()
    showCreateDialog.value = false
    newDatasetName.value = ''
  } catch (error: any) {
    ElMessage.error(error.response?.data?.detail || '创建失败')
  } finally {
    isCreating.value = false
  }
}

const isUploading = ref(false)
const uploadQueue = ref<File[]>([])

function beforeUpload(file: File) {
  const isImage = file.type.startsWith('image/')
  const isTxt = file.name.endsWith('.txt')
  const isSafetensors = file.name.endsWith('.safetensors')
  
  if (!isImage && !isTxt && !isSafetensors) {
    ElMessage.error('不支持的文件格式')
    return false
  }
  
  const isLt100M = file.size / 1024 / 1024 < 100
  if (!isLt100M) {
    ElMessage.error('文件大小不能超过 100MB')
    return false
  }
  
  uploadQueue.value.push(file)
  // Debounce upload
  setTimeout(() => processUploadQueue(), 100)
  return false // Prevent default upload
}

async function processUploadQueue() {
  if (uploadQueue.value.length === 0 || isUploading.value || !currentView.value) return
  
  const files = [...uploadQueue.value]
  uploadQueue.value = []
  
  isUploading.value = true
  try {
    const formData = new FormData()
    formData.append('dataset', currentView.value.name)
    files.forEach(file => {
      formData.append('files', file)
    })
    
    const response = await axios.post('/api/dataset/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    
    if (response.data.uploaded?.length > 0) {
      ElMessage.success(`成功上传 ${response.data.uploaded.length} 个文件`)
      await loadLocalDatasets()
      // Auto scan the dataset
      if (response.data.datasetPath) {
        datasetPath.value = response.data.datasetPath
        await datasetStore.scanDataset(response.data.datasetPath)
      }
    }
    if (response.data.errors?.length > 0) {
      ElMessage.warning(`${response.data.errors.length} 个文件上传失败`)
    }
  } catch (error: any) {
    ElMessage.error('上传失败: ' + (error.response?.data?.detail || error.message || '未知错误'))
  } finally {
    isUploading.value = false
  }
}

async function customUpload() {
  // Custom upload is handled by beforeUpload + processUploadQueue
  return Promise.resolve()
}

onMounted(async () => {
  loadLocalDatasets()
  // 检查是否有正在进行的 Ollama 标注任务
  await checkOllamaTaggingStatus()
})

// 监听缓存状态变化，完成时自动刷新数据集
watch(
  () => cacheStatus.value,
  async (newStatus, oldStatus) => {
    if (!currentView.value) return
    
    // Latent 缓存完成时刷新
    if (oldStatus?.latent?.status === 'running' && newStatus?.latent?.status === 'completed') {
      console.log('[Dataset] Latent cache completed, refreshing dataset...')
      await datasetStore.scanDataset(currentView.value.path)
    }
    
    // Text 缓存完成时刷新
    if (oldStatus?.text?.status === 'running' && newStatus?.text?.status === 'completed') {
      console.log('[Dataset] Text cache completed, refreshing dataset...')
      await datasetStore.scanDataset(currentView.value.path)
      isGeneratingCache.value = false
    }
    
    // 两个都完成，重置生成状态
    if (newStatus?.latent?.status !== 'running' && newStatus?.text?.status !== 'running') {
      if (isGeneratingCache.value) {
        isGeneratingCache.value = false
      }
    }
  },
  { deep: true }
)

// 检查并恢复 Ollama 标注状态
async function checkOllamaTaggingStatus() {
  try {
    const res = await axios.get('/api/dataset/ollama/status')
    if (res.data.running) {
      // 有正在进行的标注任务，恢复状态和轮询
      ollamaTagging.value = true
      ollamaStatus.value = res.data
      startOllamaStatusPolling()
    } else if (res.data.total > 0 && res.data.completed > 0) {
      // 有已完成的任务，显示状态
      ollamaStatus.value = res.data
    }
  } catch (e) {
    // 忽略错误，可能后端还没准备好
  }
}

// 启动状态轮询
let ollamaPollingInterval: ReturnType<typeof setInterval> | null = null

function startOllamaStatusPolling() {
  if (ollamaPollingInterval) return
  
  ollamaPollingInterval = setInterval(async () => {
    try {
      const statusRes = await axios.get('/api/dataset/ollama/status')
      ollamaStatus.value = statusRes.data
      
      if (!statusRes.data.running) {
        stopOllamaStatusPolling()
        ollamaTagging.value = false
        ElMessage.success(`标注完成！成功: ${statusRes.data.completed}`)
        // 刷新数据集
        if (currentView.value) {
          await datasetStore.scanDataset(currentView.value.path)
        }
      }
    } catch (e) {
      stopOllamaStatusPolling()
      ollamaTagging.value = false
    }
  }, 2000)
}

function stopOllamaStatusPolling() {
  if (ollamaPollingInterval) {
    clearInterval(ollamaPollingInterval)
    ollamaPollingInterval = null
  }
}

async function scanDataset() {
  if (!datasetPath.value.trim()) {
    ElMessage.warning('请输入数据集路径')
    return
  }
  
  try {
    await datasetStore.scanDataset(datasetPath.value)
    ElMessage.success(`已加载 ${datasetStore.currentDataset?.imageCount} 张图片`)
  } catch (error: any) {
    ElMessage.error(error.message || '扫描失败')
  }
}

function toggleSelection(image: DatasetImage) {
  datasetStore.toggleImageSelection(image.path)
}

// 是否全选
const isAllSelected = computed(() => {
  return datasetStore.currentImages.length > 0 && 
    datasetStore.selectedImages.size === datasetStore.currentImages.length
})

// 切换全选/取消全选
function toggleSelectAll() {
  if (isAllSelected.value) {
    datasetStore.clearSelection()
  } else {
    datasetStore.selectAll()
  }
}

// 删除选中的图片
async function deleteSelected() {
  if (datasetStore.selectedImages.size === 0) return
  
  try {
    await ElMessageBox.confirm(
      `确定要删除选中的 ${datasetStore.selectedImages.size} 张图片吗？此操作不可恢复！`,
      '确认删除',
      {
        confirmButtonText: '删除',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    const paths = Array.from(datasetStore.selectedImages)
    const response = await axios.post('/api/dataset/delete-images', { paths })
    
    if (response.data.deleted > 0) {
      ElMessage.success(`成功删除 ${response.data.deleted} 张图片`)
      datasetStore.clearSelection()
      // 重新扫描数据集
      if (currentView.value) {
        await datasetStore.scanDataset(currentView.value.path)
      }
    }
    if (response.data.errors?.length > 0) {
      ElMessage.warning(`${response.data.errors.length} 张图片删除失败`)
    }
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败: ' + (error.response?.data?.detail || error.message || '未知错误'))
    }
  }
}

async function generateCaptions() {
  isGenerating.value = true
  try {
    await datasetStore.generateCaptions(captionModel.value as 'qwen' | 'blip')
    ElMessage.success('标注生成完成')
    showCaptionDialog.value = false
    // 重新扫描以更新标注
    await datasetStore.scanDataset(datasetPath.value)
  } catch (error: any) {
    ElMessage.error(error.message || '生成失败')
  } finally {
    isGenerating.value = false
  }
}

// 图片缩放方法
const resizeProgress = computed(() => {
  if (resizeStatus.value.total === 0) return 0
  return Math.round((resizeStatus.value.completed / resizeStatus.value.total) * 100)
})

async function confirmResize() {
  if (!currentView.value) return
  
  try {
    await ElMessageBox.confirm(
      '此操作将直接覆盖原图，不可撤销！确定要继续吗？',
      '警告',
      {
        confirmButtonText: '确认缩放',
        cancelButtonText: '取消',
        type: 'warning',
        confirmButtonClass: 'el-button--danger'
      }
    )
  } catch {
    return
  }
  
  resizing.value = true
  resizeStatus.value = { running: true, total: 0, completed: 0, current_file: '', errors: [] }
  
  try {
    const res = await axios.post('/api/dataset/resize', {
      dataset_path: currentView.value.path,
      max_long_edge: resizeConfig.value.maxLongEdge,
      quality: resizeConfig.value.quality,
      sharpen: resizeConfig.value.sharpen
    })
    
    if (res.data.total === 0) {
      ElMessage.info('没有图片需要处理')
      resizing.value = false
      return
    }
    
    ElMessage.success(`开始处理 ${res.data.total} 张图片`)
    
    // 轮询进度
    const pollInterval = setInterval(async () => {
      try {
        const statusRes = await axios.get('/api/dataset/resize/status')
        resizeStatus.value = statusRes.data
        
        if (!statusRes.data.running) {
          clearInterval(pollInterval)
          resizing.value = false
          ElMessage.success(`处理完成！共 ${statusRes.data.completed} 张`)
          // 刷新数据集
          if (currentView.value) {
            await datasetStore.scanDataset(currentView.value.path)
          }
        }
      } catch (e) {
        console.error('Poll status error:', e)
      }
    }, 500)
    
  } catch (e: any) {
    ElMessage.error('启动失败: ' + (e.response?.data?.detail || e.message))
    resizing.value = false
  }
}

async function stopResize() {
  try {
    await axios.post('/api/dataset/resize/stop')
    ElMessage.info('正在停止...')
  } catch (e) {
    console.error('Stop error:', e)
  }
}

// 删除所有标注
async function confirmDeleteCaptions() {
  if (!currentView.value) return
  
  try {
    await ElMessageBox.confirm(
      '确定要删除该数据集中所有 .txt 标注文件吗？此操作不可撤销！',
      '删除标注',
      {
        confirmButtonText: '确认删除',
        cancelButtonText: '取消',
        type: 'warning',
        confirmButtonClass: 'el-button--danger'
      }
    )
  } catch {
    return
  }
  
  try {
    const res = await axios.post('/api/dataset/delete-captions', {
      dataset_path: currentView.value.path
    })
    
    if (res.data.deleted > 0) {
      ElMessage.success(`成功删除 ${res.data.deleted} 个标注文件`)
      // 刷新数据集
      await datasetStore.scanDataset(currentView.value.path)
    } else {
      ElMessage.info('没有标注文件需要删除')
    }
    
    if (res.data.errors?.length > 0) {
      ElMessage.warning(`${res.data.errors.length} 个文件删除失败`)
    }
  } catch (e: any) {
    ElMessage.error('删除失败: ' + (e.response?.data?.detail || e.message))
  }
}

// Ollama 相关方法
const ollamaProgress = computed(() => {
  if (ollamaStatus.value.total === 0) return 0
  return Math.round((ollamaStatus.value.completed / ollamaStatus.value.total) * 100)
})

const canStartOllama = computed(() => {
  return ollamaConfig.value.url && ollamaConfig.value.model && currentView.value
})

async function loadOllamaModels() {
  if (ollamaModels.value.length > 0) return
  await testOllamaConnection()
}

async function testOllamaConnection() {
  testingConnection.value = true
  try {
    const res = await axios.get(`/api/dataset/ollama/models?ollama_url=${encodeURIComponent(ollamaConfig.value.url)}`)
    if (res.data.success) {
      ollamaModels.value = res.data.models
      if (res.data.models.length > 0 && !ollamaConfig.value.model) {
        // 优先选择视觉模型
        const visionModel = res.data.models.find((m: string) => 
          m.includes('llava') || m.includes('vision') || m.includes('vl')
        )
        ollamaConfig.value.model = visionModel || res.data.models[0]
      }
      ElMessage.success(`连接成功，发现 ${res.data.models.length} 个模型`)
    } else {
      ElMessage.error(res.data.error || '连接失败')
    }
  } catch (e: any) {
    ElMessage.error('连接失败: ' + (e.message || '未知错误'))
  } finally {
    testingConnection.value = false
  }
}

async function startOllamaTagging() {
  if (!currentView.value) return
  
  ollamaTagging.value = true
  ollamaStatus.value = { running: true, total: 0, completed: 0, current_file: '', errors: [] }
  
  try {
    const res = await axios.post('/api/dataset/ollama/tag', {
      dataset_path: currentView.value.path,
      ollama_url: ollamaConfig.value.url,
      model: ollamaConfig.value.model,
      prompt: ollamaConfig.value.prompt,
      max_long_edge: ollamaConfig.value.maxLongEdge,
      skip_existing: ollamaConfig.value.skipExisting,
      trigger_word: ollamaConfig.value.triggerWord,
      enable_think: ollamaConfig.value.enableThink
    })
    
    if (res.data.total === 0) {
      ElMessage.info('没有需要标注的图片')
      ollamaTagging.value = false
      return
    }
    
    ElMessage.success(`开始标注 ${res.data.total} 张图片`)
    ollamaStatus.value.total = res.data.total
    
    // 使用统一的轮询函数
    startOllamaStatusPolling()
    
  } catch (e: any) {
    ElMessage.error('启动失败: ' + (e.response?.data?.detail || e.message))
    ollamaTagging.value = false
  }
}

async function stopOllamaTagging() {
  try {
    await axios.post('/api/dataset/ollama/stop')
    ElMessage.info('正在停止...')
    stopOllamaStatusPolling()
    ollamaTagging.value = false
  } catch (e) {
    console.error('Stop error:', e)
  }
}

function formatSize(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}
</script>

<style lang="scss" scoped>
.dataset-page {
  max-width: 1600px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: var(--space-xl);
  
  h1 {
    font-family: var(--font-display);
    font-size: 2rem;
    margin-bottom: var(--space-xs);
  }
  
  .subtitle {
    color: var(--text-muted);
    font-size: 13px;
  }
  
}

// 详情页顶栏
.detail-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-md) var(--space-lg);
  margin-bottom: var(--space-lg);
  gap: var(--space-lg);
  
  .header-left {
    display: flex;
    align-items: center;
    gap: var(--space-md);
    min-width: 0;
    flex: 1;
    
    .back-btn {
      flex-shrink: 0;
      width: 40px;
      height: 40px;
      padding: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .header-info {
      min-width: 0;
      
      h2 {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      
      .path-text {
        font-size: 12px;
        color: var(--text-muted);
        display: block;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
    }
  }
  
  .header-right {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    flex-shrink: 0;
  }
}

// 文件夹网格
.folder-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: var(--space-lg);
}

.folder-card {
  padding: var(--space-lg);
  cursor: pointer;
  transition: all var(--transition-fast);
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  
  &:hover {
    transform: translateY(-4px);
    border-color: var(--primary);
    
    .delete-btn {
      opacity: 1;
    }
  }
  
  .folder-icon {
    color: var(--primary);
    margin-bottom: var(--space-md);
  }
  
  .folder-info {
    .folder-name {
      font-weight: 600;
      font-size: 15px;
      margin-bottom: var(--space-xs);
      word-break: break-all;
    }
    
    .folder-meta {
      font-size: 13px;
      color: var(--text-muted);
    }
  }
  
  .delete-btn {
    position: absolute;
    top: var(--space-sm);
    right: var(--space-sm);
    opacity: 0;
    transition: opacity var(--transition-fast);
  }
}

.dataset-toolbar {
  padding: var(--space-lg);
  margin-bottom: var(--space-lg);
  display: flex;
  align-items: center;
  gap: var(--space-md);
  flex-wrap: wrap;
  
  .el-divider--vertical {
    height: 32px;
    margin: 0;
  }
  
  .toolbar-section {
    display: flex;
    flex-direction: column;
    gap: var(--space-sm);
    
    .section-title {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
  }
  
  .input-group {
    display: flex;
    gap: var(--space-sm);
    
    .el-input {
      width: 300px;
    }
  }
  
  .upload-hint {
    font-size: 11px;
    color: var(--text-muted);
  }
}

.dataset-info {
  padding: var(--space-lg);
  margin-bottom: var(--space-lg);
  
  .info-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-md);
  }
  
  .info-stats {
    display: flex;
    gap: var(--space-xl);
    
    .stat {
      display: flex;
      align-items: center;
      gap: var(--space-sm);
      color: var(--text-secondary);
      
      .el-icon {
        color: var(--primary);
      }
      
      &.stat-success {
        color: var(--success);
        
        .el-icon {
          color: var(--success);
        }
      }
    }
  }
  
  .info-actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-sm);
    
    .el-button {
      margin: 0 !important;  /* 覆盖 Element Plus 默认 margin */
    }
  }
  
  .cache-progress-section {
    margin-top: var(--space-lg);
    padding-top: var(--space-md);
    border-top: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    gap: var(--space-md);
  }
  
  .cache-progress-item {
    .progress-label {
      display: flex;
      align-items: center;
      gap: var(--space-sm);
      margin-bottom: var(--space-xs);
      font-size: 13px;
      color: var(--text-secondary);
      
      .el-icon {
        color: var(--primary);
      }
      
      .progress-count {
        margin-left: auto;
        font-family: var(--font-mono);
        color: var(--text-primary);
      }
    }
    
    &.queued {
      opacity: 0.7;
      .el-icon {
        color: var(--text-muted);
      }
    }
  }
  
  .spinning {
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: var(--space-md);
}

.image-card {
  padding: var(--space-sm);
  cursor: pointer;
  transition: all var(--transition-fast);
  position: relative;
  display: flex;
  flex-direction: column;
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
  }
  
  &.selected {
    border-color: var(--primary);
    box-shadow: 0 0 20px var(--primary-glow);
    
    .image-overlay {
      opacity: 1;
    }
  }
  
  .image-wrapper {
    position: relative;
    border-radius: var(--radius-md);
    overflow: hidden;
    aspect-ratio: 1;
    background: var(--bg-darker);
    cursor: pointer;
    
    img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      transition: transform var(--transition-fast);
    }
    
    .image-error-overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.85);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 8px;
      color: var(--el-color-warning);
      font-size: 12px;
      z-index: 5;
      
      .el-icon {
        font-size: 32px;
      }
      
      .el-button {
        margin-top: 4px;
      }
    }
    
    &:hover img {
      transform: scale(1.05);
    }
  }
  
  .select-circle {
    position: absolute;
    top: var(--space-sm);
    right: var(--space-sm);
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: 2px solid rgba(255, 255, 255, 0.6);
    background: rgba(0, 0, 0, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all var(--transition-fast);
    z-index: 2;
    
    &:hover {
      border-color: var(--primary);
      background: rgba(0, 0, 0, 0.6);
    }
    
    &.checked {
      border-color: var(--primary);
      background: var(--primary);
      
      .el-icon {
        color: var(--bg-dark);
        font-size: 14px;
        font-weight: bold;
      }
    }
  }
  
  .cache-tags {
    position: absolute;
    top: var(--space-xs);
    left: var(--space-xs);
    display: flex;
    gap: 4px;
  }
  
  .cache-tag {
    display: flex;
    align-items: center;
    gap: 2px;
    padding: 4px 8px;
    border-radius: var(--radius-sm);
    background: rgba(0, 0, 0, 0.7);
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 600;
    backdrop-filter: blur(4px);
    
    .el-icon {
      font-size: 12px;
    }
    
    &.active {
      background: var(--success);
      color: white;
      box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
    }
  }
  
  .image-info {
    padding: var(--space-sm);
    
    .image-name {
      font-size: 0.85rem;
      font-weight: 500;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-bottom: var(--space-xs);
    }
    
    .image-meta {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-bottom: var(--space-xs);
    }
    
    .image-caption {
      font-size: 0.75rem;
      color: var(--text-secondary);
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      
      &.no-caption {
        color: var(--text-muted);
        font-style: italic;
      }
    }
  }
  
  .image-actions {
    display: flex;
    gap: var(--space-sm);
    padding: var(--space-sm);
    padding-top: 0;
    
    .action-btn {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
      padding: 8px 12px;
      font-size: 12px;
      background: var(--bg-darker);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      color: var(--text-secondary);
      cursor: pointer;
      transition: all var(--transition-fast);
      
      .el-icon {
        font-size: 14px;
      }
      
      &:hover {
        background: var(--primary);
        border-color: var(--primary);
        color: var(--bg-dark);
      }
    }
  }
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: calc(var(--space-xl) * 3);
  text-align: center;
  min-height: 300px;
  
  .el-icon {
    color: var(--el-color-primary);
    margin-bottom: var(--space-lg);
    font-size: 64px;
  }
  
  .is-loading {
    animation: rotate 1.5s linear infinite;
  }
  
  h3 {
    margin-bottom: var(--space-sm);
    color: var(--text-primary);
    font-size: 1.25rem;
  }
  
  p {
    color: var(--text-muted);
    font-size: 0.9rem;
  }
}

@keyframes rotate {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: calc(var(--space-xl) * 2);
  text-align: center;
  
  .el-icon {
    color: var(--text-muted);
    margin-bottom: var(--space-md);
  }
  
  h3 {
    margin-bottom: var(--space-sm);
    color: var(--text-secondary);
  }
  
  p {
    color: var(--text-muted);
  }
}

.caption-edit {
  .preview-image {
    width: 200px;
    height: 200px;
    margin: 0 auto var(--space-lg);
    border-radius: var(--radius-md);
    overflow: hidden;
    
    img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
  }
}

.preview-container {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 400px;
  
  img {
    max-width: 100%;
    max-height: 80vh;
    object-fit: contain;
  }
}

.cache-path-hint {
  color: var(--text-muted);
  font-size: 12px;
  margin-left: 4px;
}

.cache-path-missing {
  color: var(--error);
  font-size: 12px;
  margin-left: 4px;
}

.cache-warning {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-md);
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.3);
  border-radius: var(--radius-md);
  color: var(--warning);
  font-size: 13px;
  margin-top: var(--space-md);
  
  .el-icon {
    font-size: 16px;
  }
  
  .el-button {
    margin-left: auto;
  }
}

.cache-hint {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-md);
  background: var(--bg-hover);
  border-radius: var(--radius-md);
  font-size: 13px;
  color: var(--text-muted);
  margin-top: var(--space-md);
  
  .el-icon {
    color: var(--info);
  }
}

/* Preview & Edit Dialog Styles */
.preview-edit-dialog :deep(.el-dialog__body) {
  padding: 24px;
  overflow: auto; /* 添加滚动支持 */
  max-height: 80vh; /* 限制最大高度 */
}

.preview-edit-layout {
  display: flex;
  flex-direction: column; /* 小屏幕时垂直排列 */
  height: 70vh;
  gap: 24px;
  overflow: hidden;
}

/* 大屏幕时使用水平布局 */
@media (min-width: 1024px) {
  .preview-edit-layout {
    flex-direction: row;
  }
}

.preview-side {
  flex: 1;
  min-height: 300px; /* 最小高度保证预览区可见 */
  display: flex;
  flex-direction: column;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
}

.preview-side .image-wrapper {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  background-image: 
    linear-gradient(45deg, #1a1a1a 25%, transparent 25%), 
    linear-gradient(-45deg, #1a1a1a 25%, transparent 25%), 
    linear-gradient(45deg, transparent 75%, #1a1a1a 75%), 
    linear-gradient(-45deg, transparent 75%, #1a1a1a 75%);
  background-size: 20px 20px;
  background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
  background-color: #111;
}

.preview-side .image-wrapper img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.image-meta-info {
  height: 40px;
  background: #1a1a1a;
  display: flex;
  align-items: center;
  justify-content: space-around;
  padding: 0 16px;
  color: #888;
  font-size: 12px;
  font-family: monospace;
  border-top: 1px solid #333;
  overflow: hidden;
  text-overflow: ellipsis;
}

.edit-side {
  flex: 1;
  min-width: 300px; /* 最小宽度保证可用性 */
  max-width: 100%; /* 确保不超出容器 */
  display: flex;
  flex-direction: column;
  background: var(--el-bg-color);
  border-radius: 8px;
  padding: 20px;
  border: 1px solid var(--el-border-color-light);
  min-height: 300px; /* 最小高度保证编辑区可用 */
}

.edit-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.edit-header h3 {
  margin: 0;
  font-size: 16px;
  font-weight: bold;
}

.caption-textarea {
  flex: 1;
  min-height: 200px; /* 最小高度保证编辑区可用性 */
}

.caption-textarea :deep(.el-textarea__inner) {
  height: 100%;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 14px;
  line-height: 1.6;
  padding: 12px;
  resize: none; /* 禁用手动调整大小，由flex控制 */
  overflow-y: auto; /* 确保文本过多时可以滚动 */
}

/* 确保对话框本身有最大宽度限制 */
.preview-edit-dialog :deep(.el-dialog) {
  max-width: 90vw; /* 最大宽度为视口的90% */
  max-height: 90vh; /* 最大高度为视口的90% */
  overflow: hidden;
}

/* 小屏幕适配 */
@media (max-width: 768px) {
  .preview-edit-dialog :deep(.el-dialog__body) {
    padding: 16px;
  }
  
  .edit-side {
    padding: 16px;
  }
  
  .preview-edit-layout {
    gap: 16px;
  }
}

.keyboard-hint {
  margin-top: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}

/* Ollama Dialog Styles */
.url-input-row {
  display: flex;
  gap: 8px;
  width: 100%;
  
  .el-input {
    flex: 1;
  }
}

.form-hint {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 4px;
}

.switch-label {
  margin-left: 8px;
  font-size: 13px;
  color: var(--text-secondary);
}

.ollama-progress {
  margin-top: 20px;
  padding: 16px;
  background: var(--bg-hover);
  border-radius: var(--radius-md);
}

.progress-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
  font-size: 12px;
  color: var(--text-muted);
  
  .error-count {
    color: var(--error);
  }
}

.resize-progress {
  margin-top: 20px;
  padding: 16px;
  background: var(--bg-hover);
  border-radius: var(--radius-md);
}

.resize-hint {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 16px;
  padding: 12px;
  background: rgba(64, 158, 255, 0.1);
  border-radius: var(--radius-md);
  font-size: 13px;
  color: var(--el-color-primary);
}

/* 分桶计算器样式 */
.bucket-dialog {
  .bucket-config {
    margin-bottom: 20px;
    padding: 16px;
    background: var(--bg-hover);
    border-radius: var(--radius-md);
  }
  
  .bucket-summary {
    display: flex;
    gap: 24px;
    margin-bottom: 20px;
    padding: 16px;
    background: var(--bg-card);
    border-radius: var(--radius-md);
    
    .summary-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      
      .label {
        font-size: 12px;
        color: var(--text-muted);
      }
      
      .value {
        font-size: 24px;
        font-weight: bold;
        color: var(--primary);
        
        &.text-warning {
          color: var(--warning);
        }
      }
    }
  }
  
  .bucket-results {
    .text-warning {
      color: var(--warning);
      font-weight: bold;
    }
  }
  
  .bucket-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    color: var(--text-muted);
    
    .el-icon {
      margin-bottom: 16px;
      opacity: 0.5;
    }
    
    p {
      font-size: 14px;
    }
  }
}

</style>

